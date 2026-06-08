"""Standalone VERBATIM games-2/6 eval for a ConvNeXt run's best checkpoint.

Mirrors stage3_improved/zeroshot_reeval_on_games_2_6.py, but builds a ConvNeXt-Tiny model.
Reuses RealGameDataset/eval_loader/metrics from rescan_checkpoint_selection.py (the harness
that reproduced ResNet s00's 0.9085 exactly), so the number is directly comparable.

Usage:
  python eval_games_2_6.py --run_name convnext_stage3 [--ckpt best_real.pt]
"""
import sys, os, json, argparse
sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny

from rescan_checkpoint_selection import (
    RealGameDataset, eval_loader, metrics, PROJECT_ROOT, HELD_OUT_GAMES, NUM_CLASSES,
    BATCH_SIZE, DEVICE,
)
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]
EXP_DIR = f"{PROJECT_ROOT}/training/convnext"

ap = argparse.ArgumentParser()
ap.add_argument("--run_name", required=True)
ap.add_argument("--ckpt", default=None, help="checkpoint filename (default: best_real.pt, "
                                             "or best_synth.pt for convnext_zeroshot).")
args, _ = ap.parse_known_args()

ckpt_name = args.ckpt or ("best_synth.pt" if "zeroshot" in args.run_name else "best_real.pt")
CKPT = f"{EXP_DIR}/checkpoints/{args.run_name}/{ckpt_name}"
OUT = f"{EXP_DIR}/results/{args.run_name}/games_2_6_eval.json"

# write-guard: output stays under convnext/, never a frozen baseline.
_abs = os.path.realpath(OUT)
assert _abs.startswith(os.path.realpath(EXP_DIR) + os.sep), f"WRITE-GUARD: {_abs} not under {EXP_DIR}"
for tok in ("zero_shot", "stage1_10", "stage2_30", "stage3_323", "stage3_improved", "stage5_combined_323"):
    assert tok not in _abs, f"WRITE-GUARD: {_abs} names frozen baseline '{tok}'"


def build_model():
    m = convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    return m


print(f"loading {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = build_model().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

all_p, all_y, per_game = [], [], {}
for N in HELD_OUT_GAMES:
    ds = RealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                         f"{PROJECT_ROOT}/data/game{N}_per_frame/images", f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    p, y = eval_loader(model, ld)
    per_game[f"game{N}"] = {"per_square": metrics(p, y)[0], "piece_only": metrics(p, y)[1]}
    all_p.append(p); all_y.append(y)
    print(f"  game{N}: {ds.manifest['image_name'].nunique()} frames, {len(p)} squares  "
          f"per-sq={per_game[f'game{N}']['per_square']:.4f}")

preds = np.concatenate(all_p); labels = np.concatenate(all_y)
persq, piece, empty = metrics(preds, labels)
per_class = {CLASS_SHORT[c]: (float((preds[labels == c] == c).mean()) if (labels == c).any() else None)
             for c in range(NUM_CLASSES)}
payload = {"model": f"ConvNeXt-Tiny {args.run_name}", "checkpoint": ckpt_name,
           "test_partition": ["game2", "game6"], "n_squares": int(len(preds)),
           "per_square_acc": persq, "piece_only_acc": piece, "empty_acc": empty,
           "per_class_acc": per_class, "per_game": per_game}
json.dump(payload, open(OUT, "w"), indent=2)
print(f"\n=== ConvNeXt {args.run_name} on games 2/6 ===")
print(f"  per-square={persq:.4f}  piece-only={piece:.4f}  empty={empty:.4f}")
print(f"wrote {OUT}")
