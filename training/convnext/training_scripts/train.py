"""
ConvNeXt-Tiny architecture comparison — ONE script, three training styles via --mode.

Mirrors our ResNet-18 runs (zero-shot synth baseline / Stage 3 sequential FT / Stage 5
combined) on a ConvNeXt-Tiny backbone. ONLY the backbone changes (+ an architecture-
appropriate AdamW/cosine recipe, intentional and logged). Data, splits, crop/warp pipeline,
the 5% forgetting probe, the game7 monitor, and the games-2/6 held-out eval are IDENTICAL
to the ResNet runs.

  --mode zeroshot : synth-only, from ImageNet. Select on synth val. Produces best_synth.pt
                    (the checkpoint --mode stage3 fine-tunes FROM).
  --mode stage3   : sequential FT from the convnext_zeroshot checkpoint on real data
                    (30 manual + game4 + game5 PGN). Select on game7 real_val.
  --mode stage5   : combined synth+real from ImageNet, WeightedRandomSampler 50/50.
                    Select on game7 real_val.

Recipe (all modes): AdamW + cosine LR + weight decay. Two-phase freeze adapted to ConvNeXt:
Phase A freezes model.features (stem + 4 stages + downsamplers), trains classifier only;
Phase B unfreezes all with discriminative LRs (head=lr_head, backbone=lr_backbone). ConvNeXt
uses LayerNorm — NO BatchNorm running stats, so the BN-freeze lever does not apply.

All outputs routed through --run_name under convnext/; the frozen baselines
(zero_shot, stage1_10, stage2_30, stage3_323, stage3_improved, stage5_combined_323) are
READ-ONLY references, protected by a hard write-guard.

Usage:
  python train.py --mode zeroshot --run_name convnext_zeroshot
  python train.py --mode stage3   --run_name convnext_stage3
  python train.py --mode stage5   --run_name convnext_stage5
"""
# %% [Cell 1 — Imports + args + seeds]
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler, Subset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.transforms import ColorJitter, RandomAffine, InterpolationMode
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from preprocessing.chess_dataset import ChessSquareDataset
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (
    warp_chessboard_image, crop_square, find_corners, ChessboardNotLocatedException,
)
# Verbatim games-2/6 eval harness (the one that reproduced ResNet s00's 0.9085 exactly).
# RealGameDataset/eval_loader/metrics are architecture-agnostic; only build_model differs,
# and we supply our own ConvNeXt build_model below.
from rescan_checkpoint_selection import (
    RealGameDataset as EvalRealGameDataset,
    eval_loader as verbatim_eval_loader,
    metrics as verbatim_metrics,
)


def _parse_args():
    p = argparse.ArgumentParser(description="ConvNeXt-Tiny — 3 training styles via --mode.")
    p.add_argument("--mode", required=True, choices=["zeroshot", "stage3", "stage5"])
    p.add_argument("--run_name", required=True, type=str,
                   help="output subdir under convnext/{checkpoints,results,plots}/")
    p.add_argument("--seed", type=int, default=42)
    # Recipe knobs (defaults filled per-mode below if left as None).
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None, help="Phase-A (head-only) epochs.")
    p.add_argument("--patience", type=int, default=None, help="early-stop patience on game7 (0=off).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=None,
                   help="Phase-B backbone LR. Default per-mode: zeroshot 1e-5; stage3/stage5 3e-5 "
                        "(the larger ConvNeXt backbone would underfit the ~20k-square real set at 1e-5).")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--zeroshot_ckpt", type=str,
                   default="/home/eladbaum/chess_project/training/convnext/checkpoints/convnext_zeroshot/best_synth.pt",
                   help="source weights for --mode stage3.")
    args, _ = p.parse_known_args()
    return args


ARGS = _parse_args()
MODE = ARGS.mode
RUN_NAME = ARGS.run_name
SEED = int(ARGS.seed)

# Per-mode recipe defaults (overridable via CLI). lr_backbone: zeroshot 1e-5 (synth is the
# easy task, overfit isn't the risk); stage3/stage5 3e-5 (the ~28M ConvNeXt backbone would
# underfit the ~20k-square real set at 1e-5 — a recipe artifact, not an architecture verdict).
_DEFAULTS = {
    "zeroshot": dict(epochs=10, warmup_epochs=1, patience=0, lr_backbone=1e-5),
    "stage3":   dict(epochs=20, warmup_epochs=2, patience=6, lr_backbone=3e-5),
    "stage5":   dict(epochs=20, warmup_epochs=2, patience=6, lr_backbone=3e-5),
}[MODE]
NUM_EPOCHS = ARGS.epochs if ARGS.epochs is not None else _DEFAULTS["epochs"]
WARMUP_EPOCHS = ARGS.warmup_epochs if ARGS.warmup_epochs is not None else _DEFAULTS["warmup_epochs"]
EARLY_STOP_PATIENCE = ARGS.patience if ARGS.patience is not None else _DEFAULTS["patience"]
BATCH_SIZE = int(ARGS.batch_size)
LR_HEAD = float(ARGS.lr_head)
LR_BACKBONE = float(ARGS.lr_backbone) if ARGS.lr_backbone is not None else _DEFAULTS["lr_backbone"]
WEIGHT_DECAY = float(ARGS.weight_decay)
SELECT_ON = "synth_val" if MODE == "zeroshot" else "game7_real_val"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] mode={MODE}  run_name={RUN_NAME}  seed={SEED}  device={DEVICE}")
print(f"[config] epochs={NUM_EPOCHS}  warmup(phaseA)={WARMUP_EPOCHS}  patience={EARLY_STOP_PATIENCE}  "
      f"batch={BATCH_SIZE}  lr_head={LR_HEAD}  lr_backbone={LR_BACKBONE}  wd={WEIGHT_DECAY}  "
      f"select_on={SELECT_ON}")
if torch.cuda.is_available():
    print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
print("\033[92m✓ Cell 1 — Imports + args + seeds — OK\033[0m")


# %% [Cell 2 — Config + write-guard]
PROJECT_ROOT = "/home/eladbaum/chess_project"

REAL_LABELS_CSV = f"{PROJECT_ROOT}/data/real_labels.csv"
REAL_IMAGES_ROOT = f"{PROJECT_ROOT}/data"
GAME7_DIR = f"{PROJECT_ROOT}/data/game7_per_frame/images"
GAME7_GT_CSV = f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv"
HELD_OUT_GAMES = [2, 6]
TRAIN_PGN_GAMES = [4, 5]

SYNTH_DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1/images"
SYNTH_MANIFEST_PATH = f"{PROJECT_ROOT}/scripts/manifest.csv"
SYNTH_CORNERS_PATH = f"{PROJECT_ROOT}/scripts/corners.json"

EXP_DIR = f"{PROJECT_ROOT}/training/convnext"
CHECKPOINTS_DIR = f"{EXP_DIR}/checkpoints/{RUN_NAME}"
RESULTS_DIR = f"{EXP_DIR}/results/{RUN_NAME}"
PLOTS_DIR = f"{EXP_DIR}/plots/{RUN_NAME}"
PREDS_DIR = f"{RESULTS_DIR}/predictions"

# --- HARD WRITE-GUARD: every output dir must resolve under convnext/ and must
# NOT name any frozen-baseline directory. Mirrors stage3_improved's guard.
_FROZEN_TOKENS = ("zero_shot", "stage1_10", "stage2_30", "stage3_323",
                  "stage3_improved", "stage5_combined_323")
for _name, _d in [("CHECKPOINTS_DIR", CHECKPOINTS_DIR), ("RESULTS_DIR", RESULTS_DIR),
                  ("PLOTS_DIR", PLOTS_DIR), ("PREDS_DIR", PREDS_DIR)]:
    _abs = os.path.realpath(_d)
    assert _abs.startswith(os.path.realpath(EXP_DIR) + os.sep), (
        f"WRITE-GUARD: {_name}={_abs} is not under convnext/ ({EXP_DIR}). Aborting.")
    for _tok in _FROZEN_TOKENS:
        assert _tok not in _abs, (
            f"WRITE-GUARD: {_name}={_abs} names a frozen-baseline path ('{_tok}'). Aborting.")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

SYNTH_MONITOR_FRAC = 0.05
NUM_WORKERS = 4
SYNTH_BATCH_FRAC = 0.5            # stage5: target synth fraction per batch
NUM_SAMPLES_PER_EPOCH = 100_000   # stage5: WeightedRandomSampler draws/epoch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGENET_MEAN_DEV = IMAGENET_MEAN.to(DEVICE)
IMAGENET_STD_DEV = IMAGENET_STD.to(DEVICE)

NUM_CLASSES = 13
CLASS_NAMES = ["White Pawn", "White Rook", "White Knight", "White Bishop", "White Queen",
               "White King", "Black Pawn", "Black Rook", "Black Knight", "Black Bishop",
               "Black Queen", "Black King", "Empty"]
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]

assert "dataset_v1.5" not in SYNTH_DATASET_DIR, "use dataset_v1, not v1.5"
print(f"checkpoints: {CHECKPOINTS_DIR}\nresults:     {RESULTS_DIR}\nplots:       {PLOTS_DIR}")
print("\033[92m✓ Cell 2 — Config + write-guard — OK\033[0m")


# %% [Cell 3 — Augmentation (per-mode, ported verbatim from the ResNet recipes)]
# zero-shot: color jitter only (ResNet zero-shot params 0.2/0.2/0.2/0.05).
# stage3/stage5: stronger — jitter @0.7 -> shear @0.8 (±8°) -> noise @0.5 (std=0.015).
ZS_COLOR_JITTER = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

COLOR_JITTER_APPLY_PROB = 0.7
SHEAR_APPLY_PROB = 0.8
NOISE_APPLY_PROB = 0.5
NOISE_STD = 0.015
FT_COLOR_JITTER = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
FT_AFFINE_SHEAR = RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05),
                               shear=(-8.0, 8.0, -8.0, 8.0),
                               interpolation=InterpolationMode.BILINEAR, fill=0)


def zeroshot_transform(crop_rgb_uint8):
    """HWC uint8 RGB -> HWC uint8 RGB. Color jitter only (mirrors ResNet zero-shot aug)."""
    return np.array(ZS_COLOR_JITTER(Image.fromarray(crop_rgb_uint8)))


def ft_transform(crop_rgb_uint8):
    """HWC uint8 RGB -> HWC uint8 RGB. jitter@0.7 -> shear@0.8 -> noise@0.5 (Stage 3/5)."""
    img = Image.fromarray(crop_rgb_uint8)
    if random.random() < COLOR_JITTER_APPLY_PROB:
        img = FT_COLOR_JITTER(img)
    if random.random() < SHEAR_APPLY_PROB:
        img = FT_AFFINE_SHEAR(img)
    x = np.array(img)
    if random.random() < NOISE_APPLY_PROB:
        noise = np.random.normal(0, NOISE_STD * 255.0, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return x


TRAIN_TRANSFORM = zeroshot_transform if MODE == "zeroshot" else ft_transform
print(f"[aug] train transform = {'zeroshot (jitter only)' if MODE=='zeroshot' else 'FT (jitter+shear+noise)'}")
print("\033[92m✓ Cell 3 — Augmentation — OK\033[0m")


# %% [Cell 4 — Real-image datasets (ported verbatim from stage3_improved Cell 4/5)]
class ManualLabelsDataset(Dataset):
    """Real training set from data/real_labels.csv. image_name is relative to data/.
    Per-image find_corners + OOB fallback + corner caching. One sample per (frame × square)."""
    CORNER_OOB_TOLERANCE = 8

    def __init__(self, manifest_df, images_root, transform=None):
        self.images_root = Path(images_root)
        self.transform = transform
        rows = []
        for _, r in manifest_df.iterrows():
            game_key = r["game"]
            grid = fen_to_label_grid(r["fen"], game_key)
            for br in range(8):
                for bc in range(8):
                    rows.append({"image_name": r["image_name"], "game": game_key,
                                 "board_row": br, "board_col": bc,
                                 "label": int(grid[br, bc]), "fen": r["fen"]})
        self.manifest = pd.DataFrame(rows).sort_values(
            ["image_name", "board_row", "board_col"]).reset_index(drop=True)
        self._corner_cache = {}

    def __len__(self):
        return len(self.manifest)

    def _get_corners(self, image_name, bgr):
        if image_name in self._corner_cache:
            return self._corner_cache[image_name]
        H, W = bgr.shape[:2]
        try:
            np.random.seed(SEED)
            corners = find_corners(bgr)
            lo, hi_x, hi_y = -self.CORNER_OOB_TOLERANCE, W + self.CORNER_OOB_TOLERANCE, H + self.CORNER_OOB_TOLERANCE
            if not bool(np.all((corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                               & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y))):
                raise ChessboardNotLocatedException("corners OOB")
        except Exception:
            corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        self._corner_cache[image_name] = corners
        return corners

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        bgr = cv2.imread(str(self.images_root / row["image_name"]))
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read {self.images_root / row['image_name']}")
        corners = self._get_corners(row["image_name"], bgr)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, int(row["board_row"]), int(row["board_col"]))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)
        return (torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0,
                int(row["label"]))


class RealGameDataset(Dataset):
    """Per-frame × per-square dataset for one full game's gt.csv (verbatim stage3 Cell 5)."""
    CORNER_OOB_TOLERANCE = 8

    def __init__(self, gt_csv_path, images_dir, game_name, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.game_name = game_name
        rows = []
        with open(gt_csv_path) as f:
            for r in csv.DictReader(f):
                grid = fen_to_label_grid(r["fen"], game_name)
                for br in range(8):
                    for bc in range(8):
                        rows.append({"image_name": r["image_name"], "board_row": br,
                                     "board_col": bc, "label": int(grid[br, bc]), "fen": r["fen"]})
        self.manifest = pd.DataFrame(rows).sort_values(
            ["image_name", "board_row", "board_col"]).reset_index(drop=True)
        self._corner_cache = {}

    def __len__(self):
        return len(self.manifest)

    def _get_corners(self, image_name, bgr):
        if image_name in self._corner_cache:
            return self._corner_cache[image_name]
        H, W = bgr.shape[:2]
        try:
            np.random.seed(SEED)
            corners = find_corners(bgr)
            lo, hi_x, hi_y = -self.CORNER_OOB_TOLERANCE, W + self.CORNER_OOB_TOLERANCE, H + self.CORNER_OOB_TOLERANCE
            if not bool(np.all((corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                               & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y))):
                raise ChessboardNotLocatedException("corners OOB")
        except Exception:
            corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        self._corner_cache[image_name] = corners
        return corners

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        bgr = cv2.imread(str(self.images_dir / row["image_name"]))
        corners = self._get_corners(row["image_name"], bgr)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, int(row["board_row"]), int(row["board_col"]))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)
        return (torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0,
                int(row["label"]))


print("\033[92m✓ Cell 4 — Dataset classes — OK\033[0m")


# %% [Cell 5 — Build datasets per mode + the 5% forgetting-probe slice]
# game7 monitor (real_val): used for selection in stage3/stage5, logged-only in zeroshot.
real_val_dataset = RealGameDataset(GAME7_GT_CSV, GAME7_DIR, game_name="game7", transform=None)
print(f"game7 real_val: {len(real_val_dataset):,} squares "
      f"({real_val_dataset.manifest['image_name'].nunique()} frames)")

# 5% slice of dataset_v1 — the forgetting probe (identical selection to the ResNet runs).
synth_manifest = pd.read_csv(SYNTH_MANIFEST_PATH)
unique_synth_imgs = sorted(synth_manifest["source_image"].unique())
slice_rng = random.Random(SEED)
_shuf = list(unique_synth_imgs)
slice_rng.shuffle(_shuf)
n_slice = max(1, int(len(_shuf) * SYNTH_MONITOR_FRAC))
slice_imgs = set(_shuf[:n_slice])
synth_monitor_df = synth_manifest[synth_manifest["source_image"].isin(slice_imgs)].reset_index(drop=True)
synth_monitor_dataset = ChessSquareDataset(synth_monitor_df, SYNTH_CORNERS_PATH,
                                           dataset_dir=SYNTH_DATASET_DIR, transform=None)
print(f"synth_monitor (5% slice, seed={SEED}): {n_slice} images, {len(synth_monitor_dataset):,} squares")

synth_val_dataset = None  # only built for zeroshot (selection signal)

if MODE == "zeroshot":
    # Full synth, split 90/10 BY SOURCE-IMAGE (project rule; cleaner than the ResNet
    # crop-level split — only affects synth-val checkpoint selection, not the 2/6 number).
    split_rng = random.Random(SEED)
    imgs = list(unique_synth_imgs)
    split_rng.shuffle(imgs)
    n_val = int(0.1 * len(imgs))
    val_imgs = set(imgs[:n_val])
    train_df = synth_manifest[~synth_manifest["source_image"].isin(val_imgs)].reset_index(drop=True)
    val_df = synth_manifest[synth_manifest["source_image"].isin(val_imgs)].reset_index(drop=True)
    train_dataset = ChessSquareDataset(train_df, SYNTH_CORNERS_PATH,
                                       dataset_dir=SYNTH_DATASET_DIR, transform=TRAIN_TRANSFORM)
    synth_val_dataset = ChessSquareDataset(val_df, SYNTH_CORNERS_PATH,
                                           dataset_dir=SYNTH_DATASET_DIR, transform=None)
    print(f"zeroshot: train={len(train_dataset):,} squares ({train_df['source_image'].nunique()} imgs)  "
          f"synth_val={len(synth_val_dataset):,} squares ({len(val_imgs)} imgs)")
    train_sampler = None
else:
    # Real training data: 30 manual (games 8-11) + full game4 + game5 PGN.
    manual_df = pd.read_csv(REAL_LABELS_CSV)
    manual_rows = [{"game_num": int(r["game"].replace("game", "")), "game": r["game"],
                    "image_name": r["image_path"], "fen": r["fen"], "view": r["view"],
                    "ply": int(r["ply"])}
                   for _, r in manual_df.sort_values(["game", "ply"]).reset_index(drop=True).iterrows()]
    stage_manual_df = pd.DataFrame(manual_rows)
    assert len(stage_manual_df) == 30, f"expected 30 manual frames, got {len(stage_manual_df)}"
    assert set(stage_manual_df["game"].unique()) == {"game8", "game9", "game10", "game11"}
    manual_train_dataset = ManualLabelsDataset(stage_manual_df, REAL_IMAGES_ROOT, transform=TRAIN_TRANSFORM)
    pgn_train_datasets = []
    for N in TRAIN_PGN_GAMES:
        ds = RealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                             f"{PROJECT_ROOT}/data/game{N}_per_frame/images",
                             game_name=f"game{N}", transform=TRAIN_TRANSFORM)
        pgn_train_datasets.append(ds)
        print(f"  PGN game{N}: {len(ds):,} squares ({ds.manifest['image_name'].nunique()} frames)")
    real_train_dataset = ConcatDataset([manual_train_dataset] + pgn_train_datasets)
    print(f"real train (manual + game4 + game5): {len(real_train_dataset):,} squares")

    if MODE == "stage3":
        train_dataset = real_train_dataset
        train_sampler = None
    else:  # stage5 — combined synth + real, WeightedRandomSampler 50/50, 100k/epoch
        synth_train_dataset = ChessSquareDataset(synth_manifest, SYNTH_CORNERS_PATH,
                                                 dataset_dir=SYNTH_DATASET_DIR, transform=TRAIN_TRANSFORM)
        n_synth = len(synth_train_dataset)
        n_real = len(real_train_dataset)
        # ConcatDataset: synth first [0, n_synth), real second [n_synth, n_synth+n_real).
        train_dataset = ConcatDataset([synth_train_dataset, real_train_dataset])
        w_per_synth = SYNTH_BATCH_FRAC / n_synth
        w_per_real = (1.0 - SYNTH_BATCH_FRAC) / n_real
        sample_weights = torch.tensor([w_per_synth] * n_synth + [w_per_real] * n_real, dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights=sample_weights,
                                              num_samples=NUM_SAMPLES_PER_EPOCH, replacement=True)
        print(f"stage5 combined: synth={n_synth:,} + real={n_real:,}; sampler 50/50, "
              f"{NUM_SAMPLES_PER_EPOCH:,} draws/epoch "
              f"(~{NUM_SAMPLES_PER_EPOCH*0.5/n_real:.1f}× per real square/epoch)")

print("\033[92m✓ Cell 5 — Datasets — OK\033[0m")


# %% [Cell 6 — DataLoaders]
def _worker_init_fn(worker_id):
    import random as _r
    s = SEED + worker_id
    _r.seed(s); np.random.seed(s); torch.manual_seed(s)


if train_sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                              worker_init_fn=_worker_init_fn, drop_last=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                              worker_init_fn=_worker_init_fn, drop_last=False)
# Eval loaders use persistent_workers=False: with 3-4 loaders alive at once on a 6-core box,
# persistent eval workers stack up (4 per loader) and thrash CPU. Non-persistent eval workers
# are torn down between eval passes, leaving cores free for the train loader.
real_val_loader = DataLoader(real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False,
                             worker_init_fn=_worker_init_fn)
synth_monitor_loader = DataLoader(synth_monitor_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False,
                                  worker_init_fn=_worker_init_fn)
synth_val_loader = None
if synth_val_dataset is not None:
    synth_val_loader = DataLoader(synth_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False,
                                  worker_init_fn=_worker_init_fn)
print("\033[92m✓ Cell 6 — DataLoaders — OK\033[0m")


# %% [Cell 7 — Model + freeze helpers + load source weights]
def build_model(pretrained):
    """ConvNeXt-Tiny; classifier[2] (Linear 768->1000) swapped to 768->13."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = convnext_tiny(weights=weights)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    return m


def freeze_backbone(model):
    """Phase A: freeze model.features (stem + 4 stages + downsamplers); train classifier only."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


SOURCE = {"zeroshot": "ImageNet", "stage3": ARGS.zeroshot_ckpt, "stage5": "ImageNet"}[MODE]
if MODE == "stage3":
    assert os.path.exists(ARGS.zeroshot_ckpt), (
        f"stage3 source checkpoint not found: {ARGS.zeroshot_ckpt}. Run --mode zeroshot first.")
    model = build_model(pretrained=False).to(DEVICE)
    src = torch.load(ARGS.zeroshot_ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(src["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
    print(f"[stage3] loaded convnext_zeroshot weights from {ARGS.zeroshot_ckpt} "
          f"(epoch {src.get('epoch')}, synth_val_acc={src.get('synth_val_acc', float('nan'))})")
else:
    model = build_model(pretrained=True).to(DEVICE)
    print(f"[{MODE}] built ConvNeXt-Tiny from ImageNet weights (head -> {NUM_CLASSES})")

n_total = sum(p.numel() for p in model.parameters())
print(f"[model] ConvNeXt-Tiny total params: {n_total:,}")
print("\033[92m✓ Cell 7 — Model + source weights — OK\033[0m")


# %% [Cell 8 — Helpers: normalize / train / eval]
def imagenet_normalize(x):
    return (x - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV


def train_one_epoch(model, loader, criterion, optimizer, print_every=100):
    model.train()
    total_loss = total_correct = total_count = 0
    t0 = time.perf_counter()
    for i, (xb, yb) in enumerate(loader, 1):
        xb = imagenet_normalize(xb.to(DEVICE, non_blocking=True))
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_count += bs
        if i % print_every == 0:
            print(f"    batch {i:4d}/{len(loader)}  loss={total_loss/total_count:.4f}  "
                  f"acc={total_correct/total_count:.4f}  ({time.perf_counter()-t0:.0f}s)")
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_correct = total_count = 0
    all_preds, all_labels = [], []
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = imagenet_normalize(xb.to(DEVICE, non_blocking=True))
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = crit(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        total_correct += (preds == yb).sum().item()
        total_count += bs
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    return (total_loss / max(total_count, 1), total_correct / max(total_count, 1),
            np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64),
            np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64))


def per_class_accuracy(preds, labels):
    out = []
    for c in range(NUM_CLASSES):
        m = labels == c
        out.append(float((preds[m] == c).mean()) if m.any() else float("nan"))
    return out


def confusion_matrix_np(preds, labels):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, title, save_path, cmap="Blues"):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)], rotation=45, ha="right")
    ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.colorbar(im)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="black" if cm_norm[i, j] < 0.5 else "white")
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


print("\033[92m✓ Cell 8 — Helpers — OK\033[0m")


# %% [Cell 9 — Smoke test (must pass before GPU time)]
print("=" * 64 + "\nSMOKE TEST\n" + "=" * 64)
xb, yb = next(iter(train_loader))
print(f"  train batch: x={tuple(xb.shape)} {xb.dtype} range=[{xb.min():.3f},{xb.max():.3f}]  "
      f"y={tuple(yb.shape)} {yb.dtype} range=[{int(yb.min())},{int(yb.max())}]")
assert xb.shape[1:] == (3, 100, 100) and xb.dtype == torch.float32
assert yb.dtype == torch.int64 and 0 <= int(yb.min()) and int(yb.max()) <= 12
assert torch.isfinite(xb).all()
# augmentation fires
s1, _ = train_dataset[0]; s2, _ = train_dataset[0]
aug_diff = float(np.abs(s1.numpy() - s2.numpy()).mean())
print(f"  aug fires: mean|s1-s2| reading train_dataset[0] twice = {aug_diff:.4f}")
assert aug_diff > 0.01, "augmentation not firing"
# forward pass
logits = model(imagenet_normalize(xb.to(DEVICE)))
assert logits.shape == (xb.size(0), NUM_CLASSES) and torch.isfinite(logits).all()
print(f"  forward: logits {tuple(logits.shape)} finite ✓")

# Pre-FT (before-training) eval on the loaded source weights — baseline for forgetting Δ.
print("  [pre-train eval on source weights]")
PRE_SYNTH_MONITOR_ACC = evaluate(model, synth_monitor_loader)[1]
PRE_REAL_VAL_ACC = evaluate(model, real_val_loader)[1]
print(f"    synth_monitor (5% v1) before: {PRE_SYNTH_MONITOR_ACC:.4f}")
print(f"    game7 real_val        before: {PRE_REAL_VAL_ACC:.4f}")
if MODE == "stage3":
    assert PRE_SYNTH_MONITOR_ACC > 0.95, (
        f"convnext_zeroshot scored {PRE_SYNTH_MONITOR_ACC:.4f} on its own synth slice; expected >0.95.")
else:  # zeroshot/stage5 start from ImageNet — chess-naive, ~chance
    assert 0.0 < PRE_SYNTH_MONITOR_ACC < 0.5, (
        f"ImageNet-init synth_monitor pre-train acc={PRE_SYNTH_MONITOR_ACC:.4f}; expected ~chance.")
print("Smoke test passed.")
print("\033[92m✓ Cell 9 — Smoke test — OK\033[0m")


# %% [Cell 10 — Optimizer (Phase A: head only, AdamW, no scheduler) + training loop]
criterion = nn.CrossEntropyLoss()


def make_phaseA_optimizer(model):
    freeze_backbone(model)
    head = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(head, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)


def make_phaseB_optimizer_and_sched(model, remaining_epochs):
    unfreeze_all(model)
    backbone = [p for n, p in model.named_parameters() if n.startswith("features.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(remaining_epochs, 1), eta_min=0.01 * LR_HEAD)
    return opt, sched


optimizer = make_phaseA_optimizer(model)
scheduler = None
phase_b_started = False
n_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase A: AdamW head-only ({n_head:,} trainable params) lr={LR_HEAD} wd={WEIGHT_DECAY}; no sched.")

CKPT_BEST = f"{CHECKPOINTS_DIR}/{'best_synth.pt' if MODE=='zeroshot' else 'best_real.pt'}"
CKPT_BEST_SYNTH_MONITOR = f"{CHECKPOINTS_DIR}/best_synth_monitor.pt"
CKPT_LATEST = f"{CHECKPOINTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

training_log = []
best_select_acc = -1.0
best_select_epoch = -1
best_synth_monitor_acc = -1.0
epochs_since_best = 0
stop_reason = "completed_all_epochs"
t_total = time.perf_counter()

for epoch in range(1, NUM_EPOCHS + 1):
    phase = "A" if epoch <= WARMUP_EPOCHS else "B"
    if phase == "B" and not phase_b_started:
        optimizer, scheduler = make_phaseB_optimizer_and_sched(model, NUM_EPOCHS - WARMUP_EPOCHS)
        phase_b_started = True
        print(f"[phase A->B] unfroze all; AdamW backbone lr={LR_BACKBONE}, head lr={LR_HEAD}; "
              f"cosine T_max={NUM_EPOCHS-WARMUP_EPOCHS}")

    print(f"\n{'='*64}\nEpoch {epoch}/{NUM_EPOCHS}  phase={phase}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}\n{'='*64}")
    t_ep = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

    sm_loss, sm_acc, _, _ = evaluate(model, synth_monitor_loader)
    rv_loss, rv_acc, rv_preds, rv_labels = evaluate(model, real_val_loader)
    rv_per_class = per_class_accuracy(rv_preds, rv_labels)
    sv_acc = float("nan")
    if synth_val_loader is not None:
        _, sv_acc, _, _ = evaluate(model, synth_val_loader)
    if scheduler is not None:
        scheduler.step()

    select_acc = sv_acc if MODE == "zeroshot" else rv_acc
    dt = time.perf_counter() - t_ep
    print(f"  epoch {epoch}: train={train_acc:.4f}  synth_mon={sm_acc:.4f}  game7={rv_acc:.4f}  "
          + (f"synth_val={sv_acc:.4f}  " if MODE == "zeroshot" else "")
          + f"select({SELECT_ON})={select_acc:.4f}  ({dt:.0f}s)")

    row = {"epoch": epoch, "phase": phase, "lr": optimizer.param_groups[0]["lr"],
           "train_loss": train_loss, "train_acc": train_acc,
           "synth_monitor_acc": sm_acc, "synth_val_acc": sv_acc,
           "real_val_loss": rv_loss, "real_val_acc": rv_acc, "epoch_time_s": dt}
    for short, acc in zip(CLASS_SHORT, rv_per_class):
        row[f"game7_acc_{short}"] = acc
    training_log.append(row)
    pd.DataFrame(training_log).to_csv(LOG_CSV, index=False)

    torch.save({"epoch": epoch, "phase": phase, "model_state_dict": model.state_dict(),
                "synth_monitor_acc": sm_acc, "synth_val_acc": sv_acc, "real_val_acc": rv_acc},
               CKPT_LATEST)

    if select_acc > best_select_acc:
        best_select_acc = select_acc
        best_select_epoch = epoch
        epochs_since_best = 0
        torch.save({"epoch": epoch, "phase": phase, "model_state_dict": model.state_dict(),
                    "synth_monitor_acc": sm_acc, "synth_val_acc": sv_acc, "real_val_acc": rv_acc},
                   CKPT_BEST)
        print(f"  -> NEW BEST {SELECT_ON}={select_acc:.4f} -> {CKPT_BEST}")
    else:
        epochs_since_best += 1

    if sm_acc > best_synth_monitor_acc:
        best_synth_monitor_acc = sm_acc
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "synth_monitor_acc": sm_acc, "real_val_acc": rv_acc, "monitor_only": True},
                   CKPT_BEST_SYNTH_MONITOR)

    if EARLY_STOP_PATIENCE and epochs_since_best >= EARLY_STOP_PATIENCE:
        stop_reason = f"early_stop_patience_{EARLY_STOP_PATIENCE} (best={best_select_acc:.4f} @ ep{best_select_epoch})"
        print(f"[early stop] {SELECT_ON} no improvement in {EARLY_STOP_PATIENCE} epochs.")
        break

total_train_time = time.perf_counter() - t_total
print(f"\nTraining done. {len(training_log)} epochs in {total_train_time/60:.1f} min. {stop_reason}")
print(f"Best {SELECT_ON}={best_select_acc:.4f} @ epoch {best_select_epoch}")
print("\033[92m✓ Cell 10 — Training loop — OK\033[0m")


# %% [Cell 11 — Load best checkpoint for end-of-run evaluation]
best_ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
model = build_model(pretrained=False).to(DEVICE)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
print(f"[eval] loaded best checkpoint (epoch {best_ckpt['epoch']}, {SELECT_ON}={best_select_acc:.4f})")


# %% [Cell 12 — Training curves]
log_df = pd.read_csv(LOG_CSV)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], "-o", ms=4, label="train")
ax1.plot(log_df["epoch"], log_df["synth_monitor_acc"], "-s", ms=4, label="synth_monitor (5% v1)")
ax1.plot(log_df["epoch"], log_df["real_val_acc"], "--^", lw=2, ms=4, label="game7 real_val")
if MODE == "zeroshot":
    ax1.plot(log_df["epoch"], log_df["synth_val_acc"], "-d", ms=4, label="synth_val (selection)")
ax1.axvline(WARMUP_EPOCHS + 0.5, color="r", ls=":", alpha=0.6, label="phase A->B")
ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_ylim(-0.02, 1.02)
ax1.legend(loc="lower right"); ax1.grid(alpha=0.3); ax1.set_title(f"ConvNeXt {MODE} — accuracy")
ax2.plot(log_df["epoch"], log_df["train_loss"], "-o", ms=4, label="train")
ax2.plot(log_df["epoch"], log_df["real_val_loss"], "--^", lw=2, ms=4, label="game7 real_val")
ax2.set_xlabel("epoch"); ax2.set_ylabel("loss"); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_title("loss")
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/training_curves.png", dpi=120); plt.close()
print(f"wrote {PLOTS_DIR}/training_curves.png")


# %% [Cell 13 — Forgetting probe (synth_monitor 5% slice, Δ vs source weights)]
sm_loss, sm_acc, sm_preds, sm_labels = evaluate(model, synth_monitor_loader)
sm_per_class = per_class_accuracy(sm_preds, sm_labels)
piece_mask = sm_labels != 12
sm_piece = float((sm_preds[piece_mask] == sm_labels[piece_mask]).mean()) if piece_mask.any() else float("nan")
plot_confusion_matrix(confusion_matrix_np(sm_preds, sm_labels),
                      f"synth_monitor (5% v1) — {MODE} acc={sm_acc:.4f}",
                      f"{PLOTS_DIR}/synth_monitor_cm.png", cmap="Blues")
# Interpretation: stage3 forgetting = how much synth ability is lost after real FT.
# zeroshot/stage5 start from ImageNet (~chance), so Δ is acquisition, not forgetting — logged
# for comparability; the meaningful retention number for stage5 is vs convnext_zeroshot.
forgetting = {
    "n_samples": int(len(sm_preds)), "slice_frac": SYNTH_MONITOR_FRAC,
    "source_weights": SOURCE,
    "synth_monitor_acc_before": PRE_SYNTH_MONITOR_ACC,
    "synth_monitor_acc_after": sm_acc,
    "forgetting_delta": sm_acc - PRE_SYNTH_MONITOR_ACC,
    "piece_only_acc": sm_piece, "loss": sm_loss,
    "per_class_acc": {CLASS_SHORT[c]: sm_per_class[c] for c in range(NUM_CLASSES)},
    "note": ("Δ vs source weights. zeroshot/stage5 source=ImageNet (~chance) so Δ is "
             "acquisition not forgetting; stage3 source=convnext_zeroshot so Δ is true forgetting."),
}
Path(f"{RESULTS_DIR}/synth_monitor_results.json").write_text(json.dumps(forgetting, indent=2))
np.save(f"{PREDS_DIR}/synth_monitor_preds.npy", sm_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/synth_monitor_labels.npy", sm_labels.astype(np.int64))
print(f"forgetting Δ (synth_monitor) = {sm_acc - PRE_SYNTH_MONITOR_ACC:+.4f}  "
      f"(before {PRE_SYNTH_MONITOR_ACC:.4f} -> after {sm_acc:.4f})")


# %% [Cell 14 — game7 eval at best checkpoint]
rv_loss, rv_acc, rv_preds, rv_labels = evaluate(model, real_val_loader)
g7_per_class = per_class_accuracy(rv_preds, rv_labels)
pm = rv_labels != 12
g7_piece = float((rv_preds[pm] == rv_labels[pm]).mean()) if pm.any() else float("nan")
plot_confusion_matrix(confusion_matrix_np(rv_preds, rv_labels),
                      f"game7 (monitor) — {MODE} acc={rv_acc:.4f}", f"{PLOTS_DIR}/game7_cm.png", cmap="Reds")
game7_results = {"n_squares": int(len(rv_preds)), "per_square_acc": rv_acc,
                 "per_square_acc_before": PRE_REAL_VAL_ACC, "piece_only_acc": g7_piece, "loss": rv_loss,
                 "per_class_acc": {CLASS_SHORT[c]: g7_per_class[c] for c in range(NUM_CLASSES)}}
Path(f"{RESULTS_DIR}/game7_results.json").write_text(json.dumps(game7_results, indent=2))
np.save(f"{PREDS_DIR}/game7_preds.npy", rv_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/game7_labels.npy", rv_labels.astype(np.int64))
print(f"game7 per-square={rv_acc:.4f}  piece-only={g7_piece:.4f}")


# %% [Cell 15 — Held-out games 2/6 eval (VERBATIM harness — the comparable number)]
# Uses the imported RealGameDataset/eval_loader/metrics from rescan_checkpoint_selection.py
# (the harness that reproduced ResNet s00's 0.9085 exactly). Only build_model differs;
# we pass our ConvNeXt model into the identical crop+normalize+argmax path.
all_p, all_y, per_game = [], [], {}
for N in HELD_OUT_GAMES:
    ds = EvalRealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                             f"{PROJECT_ROOT}/data/game{N}_per_frame/images", f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    p, y = verbatim_eval_loader(model, ld)
    persq_g, piece_g, empty_g = verbatim_metrics(p, y)
    per_game[f"game{N}"] = {"n_frames": int(ds.manifest["image_name"].nunique()),
                            "n_squares": int(len(p)), "per_square": persq_g, "piece_only": piece_g}
    all_p.append(p); all_y.append(y)
    print(f"  game{N}: {ds.manifest['image_name'].nunique()} frames, {len(p)} squares  "
          f"per-sq={persq_g:.4f}  piece={piece_g:.4f}")
    plot_confusion_matrix(confusion_matrix_np(p, y), f"game{N} — {MODE} acc={persq_g:.4f}",
                          f"{PLOTS_DIR}/game{N}_cm.png", cmap="Reds")
    np.save(f"{PREDS_DIR}/game{N}_preds.npy", p.astype(np.int64))
    np.save(f"{PREDS_DIR}/game{N}_labels.npy", y.astype(np.int64))

preds = np.concatenate(all_p); labels = np.concatenate(all_y)
persq, piece, empty = verbatim_metrics(preds, labels)
held_per_class = per_class_accuracy(preds, labels)
games_2_6 = {
    "model": f"ConvNeXt-Tiny {MODE}", "run_name": RUN_NAME, "source_weights": SOURCE,
    "test_partition": [f"game{N}" for N in HELD_OUT_GAMES],
    "eval_path": "verbatim rescan_checkpoint_selection RealGameDataset/eval_loader/metrics "
                 "(same harness that reproduced ResNet s00's 0.9085)",
    "n_squares": int(len(preds)), "per_square_acc": persq, "piece_only_acc": piece, "empty_acc": empty,
    "per_class_acc": {CLASS_SHORT[c]: held_per_class[c] for c in range(NUM_CLASSES)},
    "per_game": per_game,
}
Path(f"{RESULTS_DIR}/games_2_6_eval.json").write_text(json.dumps(games_2_6, indent=2))
plot_confusion_matrix(confusion_matrix_np(preds, labels),
                      f"games 2/6 (held-out) — {MODE} per-sq={persq:.4f}",
                      f"{PLOTS_DIR}/games_2_6_cm.png", cmap="Reds")
print(f"\n=== GAMES 2/6 (held-out) — ConvNeXt {MODE} ===")
print(f"  per-square={persq:.4f}  piece-only={piece:.4f}  empty={empty:.4f}")


# %% [Cell 16 — recipe.json (reportable) + summary]
# Backbone feature-map resolution at our 100x100 input (recorded for the caveat below).
with torch.no_grad():
    _fmap = model.features(imagenet_normalize(torch.rand(1, 3, 100, 100, device=DEVICE)))
_feat_hw = f"{_fmap.shape[2]}x{_fmap.shape[3]}"
recipe = {
    "arch": "ConvNeXt-Tiny", "mode": MODE, "run_name": RUN_NAME, "seed": SEED,
    "param_count": int(n_total), "source_weights": SOURCE,
    "input_resolution": "100x100", "backbone_feature_map": f"{_feat_hw} ({_fmap.shape[2]*_fmap.shape[3]} tokens, 768 ch)",
    "resolution_caveat": ("ConvNeXt-Tiny evaluated at 100x100 for input-consistency with ResNet-18 "
                          "(the published 0.5138/0.9085/0.9160 all live on the 100x100 path); this is "
                          "well below its 224x224 pretraining resolution (3x3 vs 7x7 feature map) and "
                          "may understate its potential — a native-resolution comparison is left to future work."),
    "optimizer": "AdamW", "betas": [0.9, 0.999], "weight_decay": WEIGHT_DECAY,
    "lr_schedule": "cosine (phase B), eta_min=0.01*lr_head", "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
    "epochs": NUM_EPOCHS, "warmup_phaseA_epochs": WARMUP_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
    "batch_size": BATCH_SIZE,
    "freeze_scheme": "Phase A: freeze model.features (stem+4 stages+downsamplers), train classifier; "
                     "Phase B: unfreeze all, discriminative LRs. No BatchNorm (LayerNorm) -> BN-freeze N/A.",
    "augmentation": "color-jitter only (zeroshot)" if MODE == "zeroshot"
                    else "jitter@0.7 -> shear@0.8(±8°) -> noise@0.5(std=0.015)",
    "selection_metric": SELECT_ON, "selected_epoch": int(best_select_epoch),
    "sampler": ("WeightedRandomSampler 50/50 synth/real, 100k draws/epoch" if MODE == "stage5"
                else "shuffle (natural distribution)"),
    "data": {"zeroshot": "full dataset_v1 synth (90/10 by-image split for synth-val selection)",
             "stage3": "30 manual + game4 + game5 PGN (~323 frames real)",
             "stage5": "dataset_v1 synth + 30 manual + game4 + game5 PGN (combined)"}[MODE],
    "results": {"games_2_6_per_square": persq, "games_2_6_piece_only": piece,
                "game7_per_square": rv_acc, "forgetting_delta": sm_acc - PRE_SYNTH_MONITOR_ACC},
}
Path(f"{RESULTS_DIR}/recipe.json").write_text(json.dumps(recipe, indent=2))
print(f"wrote {RESULTS_DIR}/recipe.json")
print("\033[92m✓ All cells complete — run finished.\033[0m")
