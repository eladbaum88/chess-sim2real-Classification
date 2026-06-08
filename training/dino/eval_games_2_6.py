"""Standalone VERBATIM games-2/6 eval for a DINOv2 run's best checkpoint.

Mirrors convnext/eval_games_2_6.py: reuses RealGameDataset + metrics from
rescan_checkpoint_selection.py (the harness that reproduced ResNet s00's 0.9085), builds a
DINOv2 ViT-S/14 model, and inserts the resize-to-INPUT before ImageNet-normalize in the eval
loop (the single DINO change).

Usage:
  python eval_games_2_6.py --run_name dino_fine_tuned [--ckpt best_real.pt] [--input_size 224]
"""
import sys, os, json, argparse
sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from rescan_checkpoint_selection import (
    RealGameDataset, metrics, PROJECT_ROOT, HELD_OUT_GAMES, NUM_CLASSES, BATCH_SIZE, DEVICE,
)
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]
EMBED_DIM = 384
EXP_DIR = f"{PROJECT_ROOT}/dino"

ap = argparse.ArgumentParser()
ap.add_argument("--run_name", required=True)
ap.add_argument("--ckpt", default=None, help="default: best_synth.pt for dino_zeroshot else best_real.pt")
ap.add_argument("--input_size", type=int, default=224)
args, _ = ap.parse_known_args()
INPUT = args.input_size
assert INPUT % 14 == 0, f"--input_size must be divisible by 14; got {INPUT}"

ckpt_name = args.ckpt or ("best_synth.pt" if "zeroshot" in args.run_name else "best_real.pt")
CKPT = f"{EXP_DIR}/checkpoints/{args.run_name}/{ckpt_name}"
OUT = f"{EXP_DIR}/results/{args.run_name}/games_2_6_eval.json"

_abs = os.path.realpath(OUT)
assert _abs.startswith(os.path.realpath(EXP_DIR) + os.sep), f"WRITE-GUARD: {_abs} not under {EXP_DIR}"
for tok in ("zero_shot", "stage1_10", "stage2_30", "stage3_323", "stage3_improved",
            "stage5_combined_323", "convnext"):
    assert tok not in _abs, f"WRITE-GUARD: {_abs} names frozen baseline '{tok}'"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
RESIZE = T.Resize((INPUT, INPUT), antialias=True)


class DinoClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


def build_model():
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception:
        import timm
        backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                     pretrained=True, num_classes=0, img_size=INPUT)
    return DinoClassifier(backbone)


def prep(x):
    x = RESIZE(x.to(DEVICE))
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def dino_eval_loader(model, loader):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        preds.append(model(prep(xb)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


print(f"loading {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = build_model().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

all_p, all_y, per_game = [], [], {}
for N in HELD_OUT_GAMES:
    ds = RealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                         f"{PROJECT_ROOT}/data/game{N}_per_frame/images", f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    p, y = dino_eval_loader(model, ld)
    per_game[f"game{N}"] = {"per_square": metrics(p, y)[0], "piece_only": metrics(p, y)[1]}
    all_p.append(p); all_y.append(y)
    print(f"  game{N}: {ds.manifest['image_name'].nunique()} frames, {len(p)} squares  "
          f"per-sq={per_game[f'game{N}']['per_square']:.4f}")

preds = np.concatenate(all_p); labels = np.concatenate(all_y)
persq, piece, empty = metrics(preds, labels)
per_class = {CLASS_SHORT[c]: (float((preds[labels == c] == c).mean()) if (labels == c).any() else None)
             for c in range(NUM_CLASSES)}
payload = {"model": f"DINOv2-ViT-S/14 {args.run_name}", "checkpoint": ckpt_name, "input_size": INPUT,
           "test_partition": ["game2", "game6"], "n_squares": int(len(preds)),
           "per_square_acc": persq, "piece_only_acc": piece, "empty_acc": empty,
           "per_class_acc": per_class, "per_game": per_game}
json.dump(payload, open(OUT, "w"), indent=2)
print(f"\n=== DINOv2 {args.run_name} on games 2/6 (input {INPUT}) ===")
print(f"  per-square={persq:.4f}  piece-only={piece:.4f}  empty={empty:.4f}")
print(f"wrote {OUT}")
