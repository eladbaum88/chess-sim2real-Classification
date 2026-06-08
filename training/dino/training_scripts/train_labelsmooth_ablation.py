"""
DINOv2 ViT-S/14 — third backbone in the sim-to-real comparison. ONE script, four styles
via --mode. Mirrors convnext/training_scripts/train.py EXACTLY except for the documented
DINO changes (model, input resize, ViT freeze scheme, linprobe mode). Data, splits,
crop/warp pipeline, 5% forgetting probe, game7 monitor, games-2/6 verbatim eval are
identical to the ResNet/ConvNeXt runs.

  --mode zeroshot : synth-only, from DINOv2-pretrained. Select on synth val -> best_synth.pt.
  --mode stage3   : sequential FT from dino_zeroshot best_synth on real (30 manual+game4+game5).
  --mode stage5   : combined synth+real from DINOv2-pretrained, WeightedRandomSampler 50/50.
  --mode linprobe : backbone FROZEN all epochs, head only, on combined synth+real 50/50
                    (canonical DINOv2 usage). Select on game7.

DINO-specific vs convnext:
  * Model: DINOv2 ViT-S/14 (hub 'dinov2_vits14', timm fallback) -> CLS embedding (384) ->
    Linear(384,13).
  * Input: datasets still yield 100x100 crops (byte-identical to ResNet/ConvNeXt). A
    transforms.Resize((INPUT,INPUT), antialias=True) is applied at the model boundary,
    immediately before ImageNet-normalize, in train+eval+games-2/6. --input_size default 224
    (16x16=256 tokens); assert INPUT % 14 == 0 (ViT-S/14).
  * Freeze: Phase A freezes the whole backbone (patch_embed, 12 blocks, norm, cls_token,
    pos_embed), trains head only; Phase B unfreezes all (discriminative LRs). LayerNorm only
    (no BatchNorm) -> BN-freeze N/A. backbone LR default 1e-5 (ViT FT is fragile).

All outputs routed through --run_name under dino/; frozen baselines (incl. convnext) are
READ-ONLY, protected by a hard write-guard.

Usage:
  python train.py --mode zeroshot --run_name dino_zeroshot
  python train.py --mode stage3   --run_name dino_fine_tuned
  python train.py --mode stage5   --run_name dino_combined
  python train.py --mode linprobe --run_name dino_combined_linprob
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
import torchvision.transforms as T
from torchvision.transforms import ColorJitter, RandomAffine, InterpolationMode
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from preprocessing.chess_dataset import ChessSquareDataset
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (
    warp_chessboard_image, crop_square, find_corners, ChessboardNotLocatedException,
)
# Verbatim games-2/6 eval harness (reproduced ResNet s00's 0.9085). RealGameDataset + metrics
# are architecture-agnostic; we reuse them and supply a DINO eval loop with the resize inserted
# (rescan's RealGameDataset ignores its transform arg and yields 100x100, so the resize must
# live in the eval loop — the single documented eval change).
from rescan_checkpoint_selection import (
    RealGameDataset as EvalRealGameDataset,
    metrics as verbatim_metrics,
)


def _parse_args():
    p = argparse.ArgumentParser(description="DINOv2 ViT-S/14 — 4 training styles via --mode.")
    p.add_argument("--mode", required=True, choices=["zeroshot", "stage3", "stage5", "linprobe"])
    p.add_argument("--run_name", required=True, type=str,
                   help="output subdir under dino/{checkpoints,results,plots}/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input_size", type=int, default=224,
                   help="ViT input size; must be divisible by 14. 224 -> 16x16=256 tokens.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None, help="Phase-A (head-only) epochs.")
    p.add_argument("--patience", type=int, default=None, help="early-stop patience on game7 (0=off).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=None,
                   help="Phase-B backbone LR (default 1e-5; ViT FT is fragile). Unused for linprobe.")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--zeroshot_ckpt", type=str,
                   default="/home/eladbaum/chess_project/checkpoints/dino_zeroshot/best_synth.pt",
                   help="source weights for --mode stage3.")
    # Split flags — defaults reproduce the original stage3/stage5 split byte-identically.
    p.add_argument("--train_pgn_games", type=str, default="4,5",
                   help="comma-sep PGN games added to real training (default 4,5; new split 4,5,2).")
    p.add_argument("--val_game", type=str, default="game7",
                   help="real game used as val/selection monitor (default game7; new split game6).")
    p.add_argument("--test_games", type=str, default="2,6",
                   help="comma-sep held-out test games (default 2,6; new split 7).")
    # --- Real-only ablation flags (additive; defaults reproduce combined behavior) ---
    p.add_argument("--no_synth", action="store_true",
                   help="REAL-ONLY ablation: remove synth from training; uniform RandomSampler over "
                        "real with num_samples=NUM_SAMPLES_PER_EPOCH. Everything else unchanged.")
    p.add_argument("--output_root", type=str, default=None,
                   help="if set, write checkpoints/results/plots under {output_root}/ instead of "
                        "dino/{checkpoints,results,plots}/{run_name}/.")
    p.add_argument("--diag_game7", type=int, default=1, choices=[0, 1],
                   help="per-epoch game7 diagnostic eval (1=on, diagnostic-only; 0=off).")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps. effective batch = batch_size * grad_accum. "
                        "Use to fit large effective batch on small GPUs (LayerNorm model -> numerically "
                        "equivalent to a single large batch).")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="label smoothing for the TRAINING CrossEntropyLoss only (eval/selection crit "
                        "stays plain CE). 0.0 = byte-identical baseline; ablation uses 0.1.")
    args, _ = p.parse_known_args()
    return args


ARGS = _parse_args()
MODE = ARGS.mode
RUN_NAME = ARGS.run_name
SEED = int(ARGS.seed)
INPUT_SIZE = int(ARGS.input_size)
assert INPUT_SIZE % 14 == 0, f"--input_size must be divisible by 14 (ViT-S/14); got {INPUT_SIZE}"

# Per-mode recipe defaults (overridable via CLI). backbone LR 1e-5 (lower than convnext's
# 3e-5 — ViT fine-tuning is more fragile). linprobe: backbone frozen all epochs (no Phase B).
_DEFAULTS = {
    "zeroshot": dict(epochs=10, warmup_epochs=1, patience=0, lr_backbone=1e-5),
    "stage3":   dict(epochs=20, warmup_epochs=2, patience=6, lr_backbone=1e-5),
    "stage5":   dict(epochs=20, warmup_epochs=2, patience=6, lr_backbone=1e-5),
    "linprobe": dict(epochs=20, warmup_epochs=0, patience=6, lr_backbone=1e-5),  # backbone frozen throughout
}[MODE]
NUM_EPOCHS = ARGS.epochs if ARGS.epochs is not None else _DEFAULTS["epochs"]
WARMUP_EPOCHS = ARGS.warmup_epochs if ARGS.warmup_epochs is not None else _DEFAULTS["warmup_epochs"]
EARLY_STOP_PATIENCE = ARGS.patience if ARGS.patience is not None else _DEFAULTS["patience"]
BATCH_SIZE = int(ARGS.batch_size)
GRAD_ACCUM = max(1, int(ARGS.grad_accum))  # effective batch = BATCH_SIZE * GRAD_ACCUM
LABEL_SMOOTHING = float(ARGS.label_smoothing)  # training-CE label smoothing (0.0 = baseline)
LR_HEAD = float(ARGS.lr_head)
LR_BACKBONE = float(ARGS.lr_backbone) if ARGS.lr_backbone is not None else _DEFAULTS["lr_backbone"]
WEIGHT_DECAY = float(ARGS.weight_decay)
FREEZE_ALL = (MODE == "linprobe")            # backbone frozen for all epochs
SELECT_ON = "synth_val" if MODE == "zeroshot" else f"{ARGS.val_game}_real_val"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] mode={MODE}  run_name={RUN_NAME}  seed={SEED}  device={DEVICE}  input={INPUT_SIZE}")
print(f"[config] epochs={NUM_EPOCHS}  warmup(phaseA)={WARMUP_EPOCHS}  patience={EARLY_STOP_PATIENCE}  "
      f"batch={BATCH_SIZE}x{GRAD_ACCUM}accum(eff={BATCH_SIZE*GRAD_ACCUM})  lr_head={LR_HEAD}  lr_backbone={'frozen' if FREEZE_ALL else LR_BACKBONE}  "
      f"wd={WEIGHT_DECAY}  select_on={SELECT_ON}")
if torch.cuda.is_available():
    print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
print("\033[92m✓ Cell 1 — Imports + args + seeds — OK\033[0m")


# %% [Cell 2 — Config + write-guard]
PROJECT_ROOT = "/home/eladbaum/chess_project"

REAL_LABELS_CSV = f"{PROJECT_ROOT}/data/real_labels.csv"
REAL_IMAGES_ROOT = f"{PROJECT_ROOT}/data"
# Split driven by CLI flags; defaults (4,5 / game7 / 2,6) == original stage3/stage5 split.
TRAIN_PGN_GAMES = [int(x) for x in ARGS.train_pgn_games.split(",")]
VAL_GAME = ARGS.val_game                                   # val/selection monitor (default game7)
HELD_OUT_GAMES = [int(x) for x in ARGS.test_games.split(",")]   # held-out test (default 2,6)
VAL_GT_CSV = f"{PROJECT_ROOT}/data/{VAL_GAME}_per_frame/gt.csv"
VAL_DIR = f"{PROJECT_ROOT}/data/{VAL_GAME}_per_frame/images"

SYNTH_DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1/images"
SYNTH_MANIFEST_PATH = f"{PROJECT_ROOT}/scripts/manifest.csv"
SYNTH_CORNERS_PATH = f"{PROJECT_ROOT}/scripts/corners.json"

EXP_DIR = f"{PROJECT_ROOT}/training/dino"
if ARGS.output_root:
    # Self-contained ablation tree: {output_root}/{checkpoints,results,plots}/ (no per-run subdir).
    _ROOT = ARGS.output_root if os.path.isabs(ARGS.output_root) else f"{PROJECT_ROOT}/{ARGS.output_root}"
    CHECKPOINTS_DIR = f"{_ROOT}/checkpoints"
    RESULTS_DIR = f"{_ROOT}/results"
    PLOTS_DIR = f"{_ROOT}/plots"
else:
    CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints/{RUN_NAME}"
    RESULTS_DIR = f"{EXP_DIR}/results/{RUN_NAME}"
    PLOTS_DIR = f"{EXP_DIR}/plots/{RUN_NAME}"
PREDS_DIR = f"{RESULTS_DIR}/predictions"

# --- HARD WRITE-GUARD: every output dir must resolve under dino/ and must NOT name any
# frozen-baseline directory (incl. the completed convnext experiment). Mirrors convnext's guard.
_FROZEN_TOKENS = ("zero_shot", "stage1_10", "stage2_30", "stage3_323",
                  "stage3_improved", "stage5_combined_323", "convnext")
for _name, _d in [("CHECKPOINTS_DIR", CHECKPOINTS_DIR), ("RESULTS_DIR", RESULTS_DIR),
                  ("PLOTS_DIR", PLOTS_DIR), ("PREDS_DIR", PREDS_DIR)]:
    _abs = os.path.realpath(_d)
    assert _abs.startswith(os.path.realpath(EXP_DIR) + os.sep), (
        f"WRITE-GUARD: {_name}={_abs} is not under dino/ ({EXP_DIR}). Aborting.")
    for _tok in _FROZEN_TOKENS:
        assert _tok not in _abs, (
            f"WRITE-GUARD: {_name}={_abs} names a frozen-baseline path ('{_tok}'). Aborting.")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

SYNTH_MONITOR_FRAC = 0.05
NUM_WORKERS = 4
SYNTH_BATCH_FRAC = 0.5            # stage5/linprobe: target synth fraction per batch
NUM_SAMPLES_PER_EPOCH = 100_000   # stage5/linprobe: WeightedRandomSampler draws/epoch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGENET_MEAN_DEV = IMAGENET_MEAN.to(DEVICE)
IMAGENET_STD_DEV = IMAGENET_STD.to(DEVICE)
RESIZE = T.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True)  # inserted before normalize

NUM_CLASSES = 13
EMBED_DIM = 384
CLASS_NAMES = ["White Pawn", "White Rook", "White Knight", "White Bishop", "White Queen",
               "White King", "Black Pawn", "Black Rook", "Black Knight", "Black Bishop",
               "Black Queen", "Black King", "Empty"]
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]

assert "dataset_v1.5" not in SYNTH_DATASET_DIR, "use dataset_v1, not v1.5"
print(f"checkpoints: {CHECKPOINTS_DIR}\nresults:     {RESULTS_DIR}\nplots:       {PLOTS_DIR}")
print("\033[92m✓ Cell 2 — Config + write-guard — OK\033[0m")


# %% [Cell 3 — Augmentation (identical to convnext)]
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
    """HWC uint8 RGB -> HWC uint8 RGB. Color jitter only (resize happens at the model boundary)."""
    return np.array(ZS_COLOR_JITTER(Image.fromarray(crop_rgb_uint8)))


def ft_transform(crop_rgb_uint8):
    """HWC uint8 RGB -> HWC uint8 RGB. jitter@0.7 -> shear@0.8 -> noise@0.5 (resize at boundary)."""
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


# %% [Cell 4 — Real-image datasets (verbatim from convnext; yield 100x100 crops)]
class ManualLabelsDataset(Dataset):
    """Real training set from data/real_labels.csv. One sample per (frame × square)."""
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
    """Per-frame × per-square dataset for one full game's gt.csv (verbatim)."""
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
real_val_dataset = RealGameDataset(VAL_GT_CSV, VAL_DIR, game_name=VAL_GAME, transform=None)
print(f"{VAL_GAME} real_val (selection): {len(real_val_dataset):,} squares "
      f"({real_val_dataset.manifest['image_name'].nunique()} frames)")

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
    elif ARGS.no_synth:  # REAL-ONLY ABLATION — synth removed; uniform RandomSampler over real, 100k/epoch
        from torch.utils.data import RandomSampler
        n_real = len(real_train_dataset)
        train_dataset = real_train_dataset
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES_PER_EPOCH)
        print(f"\033[93m[REAL-ONLY ABLATION] synthetic samples in training = 0\033[0m")
        print(f"  real train squares = {n_real:,}  (manual 8-11 + PGN {TRAIN_PGN_GAMES})")
        print(f"  sampler = uniform RandomSampler over REAL, replacement=True, "
              f"num_samples={NUM_SAMPLES_PER_EPOCH:,}/epoch (epoch length pinned to match combined)")
    else:  # stage5 OR linprobe — combined synth + real, WeightedRandomSampler 50/50, 100k/epoch
        synth_train_dataset = ChessSquareDataset(synth_manifest, SYNTH_CORNERS_PATH,
                                                 dataset_dir=SYNTH_DATASET_DIR, transform=TRAIN_TRANSFORM)
        n_synth = len(synth_train_dataset)
        n_real = len(real_train_dataset)
        train_dataset = ConcatDataset([synth_train_dataset, real_train_dataset])
        w_per_synth = SYNTH_BATCH_FRAC / n_synth
        w_per_real = (1.0 - SYNTH_BATCH_FRAC) / n_real
        sample_weights = torch.tensor([w_per_synth] * n_synth + [w_per_real] * n_real, dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights=sample_weights,
                                              num_samples=NUM_SAMPLES_PER_EPOCH, replacement=True)
        print(f"{MODE} combined: synth={n_synth:,} + real={n_real:,}; sampler 50/50, "
              f"{NUM_SAMPLES_PER_EPOCH:,} draws/epoch")

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
# Eval loaders: persistent_workers=False (avoid stacking workers across loaders on a 6-core box).
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
DINO_LOAD_PATH = None  # 'hub' or 'timm', set in build_model


class DinoClassifier(nn.Module):
    """DINOv2 ViT-S/14 backbone -> CLS embedding (384) -> Linear(384, NUM_CLASSES)."""
    def __init__(self, backbone, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)            # (B, 384) CLS embedding
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


def build_model():
    """DINOv2 ViT-S/14 pretrained backbone + fresh 13-class head. Try torch.hub, fall back to timm.
    Callers that need a specific checkpoint load_state_dict afterward (weights overwritten)."""
    global DINO_LOAD_PATH
    backbone = None
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        DINO_LOAD_PATH = "hub:dinov2_vits14"
    except Exception as e:
        print(f"[load] torch.hub dinov2_vits14 failed ({type(e).__name__}: {e}); falling back to timm.")
        import timm
        backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                     pretrained=True, num_classes=0, img_size=INPUT_SIZE)
        DINO_LOAD_PATH = "timm:vit_small_patch14_dinov2.lvd142m"
    return DinoClassifier(backbone)


def freeze_backbone(model):
    """Phase A / linprobe: freeze the whole backbone (patch_embed, blocks, norm, cls_token,
    pos_embed); train the head only."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


SOURCE = {"zeroshot": "DINOv2-pretrained", "stage3": ARGS.zeroshot_ckpt,
          "stage5": "DINOv2-pretrained", "linprobe": "DINOv2-pretrained"}[MODE]
model = build_model().to(DEVICE)
if MODE == "stage3":
    assert os.path.exists(ARGS.zeroshot_ckpt), (
        f"stage3 source checkpoint not found: {ARGS.zeroshot_ckpt}. Run --mode zeroshot first.")
    src = torch.load(ARGS.zeroshot_ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(src["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
    print(f"[stage3] loaded dino_zeroshot weights from {ARGS.zeroshot_ckpt} "
          f"(epoch {src.get('epoch')}, synth_val_acc={src.get('synth_val_acc', float('nan'))})")
else:
    print(f"[{MODE}] built DINOv2 ViT-S/14 ({DINO_LOAD_PATH}) + fresh head -> {NUM_CLASSES}")

n_total = sum(p.numel() for p in model.parameters())
N_PATCHES = (INPUT_SIZE // 14) ** 2
print(f"[model] DINOv2 ViT-S/14 total params: {n_total:,}  |  {N_PATCHES} patch tokens @ {INPUT_SIZE}px")
print("\033[92m✓ Cell 7 — Model + source weights — OK\033[0m")


# %% [Cell 8 — Helpers: resize+normalize / train / eval]
def prep(x):
    """Move to device, RESIZE to INPUT (antialias) then ImageNet-normalize. The resize is the
    single DINO pipeline change vs convnext; bilinear-resize commutes with the affine normalize."""
    x = x.to(DEVICE, non_blocking=True)
    x = RESIZE(x)
    return (x - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV


def train_one_epoch(model, loader, criterion, optimizer, print_every=100, grad_accum=1):
    model.train()
    total_loss = total_correct = total_count = 0
    t0 = time.perf_counter()
    n_batches = len(loader)
    optimizer.zero_grad()
    for i, (xb, yb) in enumerate(loader, 1):
        xb = prep(xb)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        # Scale by 1/grad_accum so summed grads over grad_accum micro-batches == mean over the
        # effective batch (exact for mean-reduction CE; DINOv2 is LayerNorm-only, no batch-stat dep).
        (loss / grad_accum).backward()
        if i % grad_accum == 0 or i == n_batches:   # step every grad_accum micro-batches (+ final remainder)
            optimizer.step()
            optimizer.zero_grad()
        bs = yb.size(0)
        total_loss += loss.item() * bs              # report the TRUE (unscaled) loss
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_count += bs
        if i % print_every == 0:
            print(f"    batch {i:4d}/{n_batches}  loss={total_loss/total_count:.4f}  "
                  f"acc={total_correct/total_count:.4f}  ({time.perf_counter()-t0:.0f}s)")
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_correct = total_count = 0
    all_preds, all_labels = [], []
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = prep(xb)
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


@torch.no_grad()
def dino_eval_loader(model, loader):
    """Games-2/6 eval: verbatim crops (EvalRealGameDataset) + resize/normalize (prep) + argmax.
    Same as the convnext verbatim eval_loader but with the DINO resize inserted."""
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        preds.append(model(prep(xb)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


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
assert xb.shape[1:] == (3, 100, 100) and xb.dtype == torch.float32  # loader yields 100x100; resize in prep
assert yb.dtype == torch.int64 and 0 <= int(yb.min()) and int(yb.max()) <= 12
assert torch.isfinite(xb).all()
s1, _ = train_dataset[0]; s2, _ = train_dataset[0]
aug_diff = float(np.abs(s1.numpy() - s2.numpy()).mean())
print(f"  aug fires: mean|s1-s2| reading train_dataset[0] twice = {aug_diff:.4f}")
assert aug_diff > 0.01, "augmentation not firing"
_xp = prep(xb)
print(f"  prep -> {tuple(_xp.shape)} (resized to {INPUT_SIZE})")
assert _xp.shape[2:] == (INPUT_SIZE, INPUT_SIZE), "resize did not produce INPUT x INPUT"
logits = model(_xp)
assert logits.shape == (xb.size(0), NUM_CLASSES) and torch.isfinite(logits).all()
print(f"  forward: logits {tuple(logits.shape)} finite ✓")

print("  [pre-train eval on source weights]")
PRE_SYNTH_MONITOR_ACC = evaluate(model, synth_monitor_loader)[1]
PRE_REAL_VAL_ACC = evaluate(model, real_val_loader)[1]
print(f"    synth_monitor (5% v1) before: {PRE_SYNTH_MONITOR_ACC:.4f}")
print(f"    {VAL_GAME} real_val    before: {PRE_REAL_VAL_ACC:.4f}")
if MODE == "stage3":
    assert PRE_SYNTH_MONITOR_ACC > 0.95, (
        f"dino_zeroshot scored {PRE_SYNTH_MONITOR_ACC:.4f} on its own synth slice; expected >0.95.")
else:
    # DINOv2-pretrained + fresh random head: not chess-trained. Allow up to 0.85 (a random head
    # on frozen features can lean toward the dominant 'empty' class); only catch a mistakenly
    # loaded chess-trained checkpoint (~0.99).
    assert 0.0 < PRE_SYNTH_MONITOR_ACC < 0.85, (
        f"DINOv2-init synth_monitor pre-train acc={PRE_SYNTH_MONITOR_ACC:.4f}; expected well below trained.")
print("Smoke test passed.")
print("\033[92m✓ Cell 9 — Smoke test — OK\033[0m")


# %% [Cell 10 — Optimizer + training loop]
# ABLATION: label smoothing on the TRAINING criterion only (eval crit above stays plain CE).
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# === CONFIG-IDENTITY GATE vs dino_combined_Game6boosted (single variable = label_smoothing) ==========
# Print + assert; abort before training if anything diverges. The one consequence of grad-accum that
# must be proven equivalent to the baseline's batch-64 run is optimizer-steps-per-epoch.
def _ceil_div(a, b):
    return (a + b - 1) // b
_BASELINE_BATCH = 64                                          # combined_game6 ran batch 64
_eff_batch = BATCH_SIZE * GRAD_ACCUM
_baseline_steps = _ceil_div(NUM_SAMPLES_PER_EPOCH, _BASELINE_BATCH)   # ceil(100000/64) = 1563
_this_steps = _ceil_div(len(train_loader), GRAD_ACCUM)               # ceil(3125/2)   = 1563
print("\n\033[96m=== CONFIG-IDENTITY GATE (vs dino_combined_Game6boosted) ===\033[0m")
print(f"  [1] train-CE label_smoothing = {criterion.label_smoothing}  (ablation variable; baseline 0.0)")
print(f"      eval/selection crit = plain nn.CrossEntropyLoss() (label_smoothing=0.0) by construction")
print(f"  [2] effective batch = {BATCH_SIZE} x {GRAD_ACCUM}accum = {_eff_batch}")
print(f"  [3] optimizer steps/epoch: this_run={_this_steps}  baseline(batch64)={_baseline_steps}  "
      f"(len(train_loader)={len(train_loader)})")
print(f"  [4] grad-accum math: (loss/{GRAD_ACCUM}).backward(); step every {GRAD_ACCUM} micro-batches "
      f"(+final remainder) -> gradient magnitude == batch {_eff_batch}, not {GRAD_ACCUM}x")
print(f"  [5] combined path: WeightedRandomSampler 50/50 synth+real  (no_synth={ARGS.no_synth})")
print(f"      data: train_pgn={TRAIN_PGN_GAMES}+manual(8-11), val={VAL_GAME}, test={HELD_OUT_GAMES}; "
      f"epochs={NUM_EPOCHS}, patience={EARLY_STOP_PATIENCE}, lr_head={LR_HEAD}, lr_backbone={LR_BACKBONE}, "
      f"seed={SEED}, select_on={SELECT_ON}")
assert abs(criterion.label_smoothing - 0.1) < 1e-9, f"label_smoothing must be 0.1, got {criterion.label_smoothing}"
assert _eff_batch == _BASELINE_BATCH, f"effective batch {_eff_batch} != baseline {_BASELINE_BATCH}"
assert _this_steps == _baseline_steps, (
    f"STOP: optimizer steps/epoch {_this_steps} != combined_game6 {_baseline_steps} -> CONFOUNDED. Aborting.")
assert ARGS.no_synth is False, "label-smoothing ablation must use the COMBINED path (do NOT pass --no_synth)"
assert TRAIN_PGN_GAMES == [4, 5, 6] and VAL_GAME == "game2" and HELD_OUT_GAMES == [7], "split mismatch vs baseline"
assert NUM_EPOCHS == 20 and EARLY_STOP_PATIENCE == 0 and SEED == 42, "epochs/patience/seed mismatch vs baseline"
assert abs(LR_HEAD - 1e-4) < 1e-12 and abs(LR_BACKBONE - 1e-5) < 1e-12, "LR mismatch vs baseline"
print("  \033[92m✓ all config-identity assertions PASSED — single variable is label_smoothing.\033[0m")
# ==============================================================================================


def make_phaseA_optimizer(model):
    freeze_backbone(model)
    head = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(head, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)


def make_phaseB_optimizer_and_sched(model, remaining_epochs):
    unfreeze_all(model)
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": LR_BACKBONE},
         {"params": model.head.parameters(), "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(remaining_epochs, 1), eta_min=0.01 * LR_HEAD)
    return opt, sched


optimizer = make_phaseA_optimizer(model)
if FREEZE_ALL:
    # linprobe: head-only over ALL epochs with a cosine schedule; backbone never unfreezes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.01 * LR_HEAD)
else:
    scheduler = None
phase_b_started = False
n_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase A: AdamW head-only ({n_head:,} trainable params) lr={LR_HEAD} wd={WEIGHT_DECAY}"
      f"{' + cosine (linprobe, frozen backbone all epochs)' if FREEZE_ALL else '; no sched'}.")

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

# === DIAGNOSTIC ONLY (dino_combined_Game6boosted selection-confound study) ============
# game7 is the HELD-OUT TEST. Here it is evaluated every epoch PURELY for post-hoc
# curve analysis and a separate best-by-game7 checkpoint. It NEVER feeds selection,
# gradients, or early-stopping: select_acc stays game2_real_val, the optimizer never
# sees game7, and the early-stop counter (epochs_since_best) keys only off select_acc.
# Built once so the per-frame corner cache persists across epochs (fast after ep1).
DIAG_GAME7 = bool(ARGS.diag_game7)
DIAG_GAME7_LOADER = None
CKPT_BEST_GAME7_DIAG = f"{CHECKPOINTS_DIR}/best_game7_diag.pt"
best_game7_persq = -1.0
best_game7_epoch = -1
if DIAG_GAME7:
    DIAG_GAME7_DS = EvalRealGameDataset(
        f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv",
        f"{PROJECT_ROOT}/data/game7_per_frame/images", "game7", transform=None)
    DIAG_GAME7_LOADER = DataLoader(DIAG_GAME7_DS, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    print(f"[diag] per-epoch game7 logging ON ({len(DIAG_GAME7_DS):,} squares) — "
          f"diagnostic only, NOT in selection/gradient/early-stop.")
else:
    print("[diag] per-epoch game7 logging OFF (--diag_game7 0).")
# ==============================================================================

t_total = time.perf_counter()

for epoch in range(1, NUM_EPOCHS + 1):
    phase = "A" if (FREEZE_ALL or epoch <= WARMUP_EPOCHS) else "B"
    if not FREEZE_ALL and phase == "B" and not phase_b_started:
        optimizer, scheduler = make_phaseB_optimizer_and_sched(model, NUM_EPOCHS - WARMUP_EPOCHS)
        phase_b_started = True
        print(f"[phase A->B] unfroze all; AdamW backbone lr={LR_BACKBONE}, head lr={LR_HEAD}; "
              f"cosine T_max={NUM_EPOCHS-WARMUP_EPOCHS}")

    print(f"\n{'='*64}\nEpoch {epoch}/{NUM_EPOCHS}  phase={phase}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}\n{'='*64}")
    t_ep = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, grad_accum=GRAD_ACCUM)

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
    print(f"  epoch {epoch}: train={train_acc:.4f}  synth_mon={sm_acc:.4f}  {VAL_GAME}={rv_acc:.4f}  "
          + (f"synth_val={sv_acc:.4f}  " if MODE == "zeroshot" else "")
          + f"select({SELECT_ON})={select_acc:.4f}  ({dt:.0f}s)")

    row = {"epoch": epoch, "phase": phase, "lr": optimizer.param_groups[0]["lr"],
           "train_loss": train_loss, "train_acc": train_acc,
           "synth_monitor_acc": sm_acc, "synth_val_acc": sv_acc,
           "real_val_loss": rv_loss, "real_val_acc": rv_acc, "epoch_time_s": dt}
    for short, acc in zip(CLASS_SHORT, rv_per_class):
        row[f"{VAL_GAME}_acc_{short}"] = acc

    # --- DIAGNOSTIC ONLY: game7 held-out eval, logged for post-hoc curve analysis.
    # Runs AFTER the optimizer step; result is never read by select_acc / early-stop.
    if DIAG_GAME7:
        g7_preds, g7_labels = dino_eval_loader(model, DIAG_GAME7_LOADER)
        g7_persq, g7_piece, g7_empty = verbatim_metrics(g7_preds, g7_labels)
        row["game7_diag_per_square"] = g7_persq
        row["game7_diag_piece_only"] = g7_piece
        row["game7_diag_empty"] = g7_empty
        print(f"  [diag] game7 (NOT selected): per-sq={g7_persq:.4f}  piece-only={g7_piece:.4f}")
        if g7_persq > best_game7_persq:
            best_game7_persq = g7_persq
            best_game7_epoch = epoch
            torch.save({"epoch": epoch, "phase": phase, "model_state_dict": model.state_dict(),
                        "game7_diag_per_square": g7_persq, "game7_diag_piece_only": g7_piece,
                        "diagnostic_only": True}, CKPT_BEST_GAME7_DIAG)
            print(f"  -> [diag] NEW BEST game7 per-sq={g7_persq:.4f} @ ep{epoch} (saved best_game7_diag.pt)")

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
model = build_model().to(DEVICE)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
print(f"[eval] loaded best checkpoint (epoch {best_ckpt['epoch']}, {SELECT_ON}={best_select_acc:.4f})")


# %% [Cell 12 — Training curves]
log_df = pd.read_csv(LOG_CSV)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], "-o", ms=4, label="train")
ax1.plot(log_df["epoch"], log_df["synth_monitor_acc"], "-s", ms=4, label="synth_monitor (5% v1)")
ax1.plot(log_df["epoch"], log_df["real_val_acc"], "--^", lw=2, ms=4, label=f"{VAL_GAME} real_val (SELECTION)")
if "game7_diag_per_square" in log_df.columns:
    ax1.plot(log_df["epoch"], log_df["game7_diag_per_square"], "-P", lw=2, ms=5,
             color="tab:red", label="game7 per-sq (DIAG, not selected)")
    ax1.plot(log_df["epoch"], log_df["game7_diag_piece_only"], ":P", lw=2, ms=4,
             color="tab:purple", label="game7 piece-only (DIAG)")
    if best_game7_epoch > 0:
        ax1.axvline(best_game7_epoch, color="tab:red", ls="--", alpha=0.4, label=f"best game7 @ ep{best_game7_epoch}")
    if best_select_epoch > 0:
        ax1.axvline(best_select_epoch, color="tab:blue", ls="--", alpha=0.4, label=f"game2-selected @ ep{best_select_epoch}")
if MODE == "zeroshot":
    ax1.plot(log_df["epoch"], log_df["synth_val_acc"], "-d", ms=4, label="synth_val (selection)")
if not FREEZE_ALL:
    ax1.axvline(WARMUP_EPOCHS + 0.5, color="r", ls=":", alpha=0.6, label="phase A->B")
ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_ylim(-0.02, 1.02)
ax1.legend(loc="lower right"); ax1.grid(alpha=0.3); ax1.set_title(f"DINOv2 {MODE} — accuracy")
ax2.plot(log_df["epoch"], log_df["train_loss"], "-o", ms=4, label="train")
ax2.plot(log_df["epoch"], log_df["real_val_loss"], "--^", lw=2, ms=4, label=f"{VAL_GAME} real_val")
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
forgetting = {
    "n_samples": int(len(sm_preds)), "slice_frac": SYNTH_MONITOR_FRAC,
    "source_weights": SOURCE,
    "synth_monitor_acc_before": PRE_SYNTH_MONITOR_ACC,
    "synth_monitor_acc_after": sm_acc,
    "forgetting_delta": sm_acc - PRE_SYNTH_MONITOR_ACC,
    "piece_only_acc": sm_piece, "loss": sm_loss,
    "per_class_acc": {CLASS_SHORT[c]: sm_per_class[c] for c in range(NUM_CLASSES)},
    "note": ("Δ vs source weights. zeroshot/stage5/linprobe source=DINOv2-pretrained (~chance on "
             "synth) so Δ is acquisition not forgetting; stage3 source=dino_zeroshot so Δ is true "
             "forgetting. linprobe backbone is frozen so synth ability is fully retained."),
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
                      f"{VAL_GAME} (monitor) — {MODE} acc={rv_acc:.4f}", f"{PLOTS_DIR}/{VAL_GAME}_cm.png", cmap="Reds")
game7_results = {"n_squares": int(len(rv_preds)), "per_square_acc": rv_acc,
                 "per_square_acc_before": PRE_REAL_VAL_ACC, "piece_only_acc": g7_piece, "loss": rv_loss,
                 "val_game": VAL_GAME,
                 "per_class_acc": {CLASS_SHORT[c]: g7_per_class[c] for c in range(NUM_CLASSES)}}
Path(f"{RESULTS_DIR}/{VAL_GAME}_results.json").write_text(json.dumps(game7_results, indent=2))
np.save(f"{PREDS_DIR}/{VAL_GAME}_preds.npy", rv_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/{VAL_GAME}_labels.npy", rv_labels.astype(np.int64))
print(f"{VAL_GAME} per-square={rv_acc:.4f}  piece-only={g7_piece:.4f}")


# %% [Cell 15 — Held-out games 2/6 eval (verbatim crops + metrics; resize via dino_eval_loader)]
all_p, all_y, per_game = [], [], {}
for N in HELD_OUT_GAMES:
    ds = EvalRealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                             f"{PROJECT_ROOT}/data/game{N}_per_frame/images", f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    p, y = dino_eval_loader(model, ld)
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
    "model": f"DINOv2-ViT-S/14 {MODE}", "run_name": RUN_NAME, "source_weights": SOURCE,
    "input_size": INPUT_SIZE,
    "test_partition": [f"game{N}" for N in HELD_OUT_GAMES],
    "eval_path": "verbatim rescan RealGameDataset + metrics; resize-to-INPUT then ImageNet-normalize "
                 "in dino_eval_loader (same crops/metric as ResNet/ConvNeXt, + DINO resize)",
    "n_squares": int(len(preds)), "per_square_acc": persq, "piece_only_acc": piece, "empty_acc": empty,
    "per_class_acc": {CLASS_SHORT[c]: held_per_class[c] for c in range(NUM_CLASSES)},
    "per_game": per_game,
}
# Held-out filename: default split keeps the original "games_2_6_*" names byte-identical;
# a custom --test_games (e.g. the new split's game7) writes "heldout_gameN_*" instead.
HELDOUT_NAME = "games_2_6" if HELD_OUT_GAMES == [2, 6] else "heldout_" + "_".join(f"game{N}" for N in HELD_OUT_GAMES)
_test_label = "games 2/6" if HELD_OUT_GAMES == [2, 6] else "+".join(f"game{N}" for N in HELD_OUT_GAMES)
Path(f"{RESULTS_DIR}/{HELDOUT_NAME}_eval.json").write_text(json.dumps(games_2_6, indent=2))
plot_confusion_matrix(confusion_matrix_np(preds, labels),
                      f"{_test_label} (held-out) — {MODE} per-sq={persq:.4f}",
                      f"{PLOTS_DIR}/{HELDOUT_NAME}_cm.png", cmap="Reds")
print(f"\n=== HELD-OUT {_test_label} — DINOv2 {MODE} ===")
print(f"  per-square={persq:.4f}  piece-only={piece:.4f}  empty={empty:.4f}")

# === SELECTION-CONFOUND CHECK (only when the per-epoch game7 diagnostic ran) ==================
# Requires best_game7_diag.pt, which only exists when --diag_game7 1. Skip cleanly otherwise so a
# diag-off run does NOT crash before Cell 16 (recipe.json).
if DIAG_GAME7 and os.path.exists(CKPT_BEST_GAME7_DIAG):
    sel_g7_persq, sel_g7_piece, sel_epoch = persq, piece, int(best_ckpt["epoch"])
    _diag = torch.load(CKPT_BEST_GAME7_DIAG, map_location=DEVICE, weights_only=False)
    _m = build_model().to(DEVICE); _m.load_state_dict(_diag["model_state_dict"]); _m.eval()
    _bp, _by = dino_eval_loader(_m, DIAG_GAME7_LOADER)
    best_g7_persq, best_g7_piece, _ = verbatim_metrics(_bp, _by)
    STAGE55_BASELINE = {"per_square": 0.9849, "piece_only": 0.9689}
    confound = {
        "note": "game7 used for DIAGNOSIS only; selection metric was game2_real_val (patience disabled).",
        "selected_by": SELECT_ON,
        "game2_selected": {"epoch": sel_epoch, "game7_per_square": sel_g7_persq,
                           "game7_piece_only": sel_g7_piece},
        "best_by_game7_diag": {"epoch": int(_diag["epoch"]), "game7_per_square": best_g7_persq,
                               "game7_piece_only": best_g7_piece},
        "stage5_5_baseline": STAGE55_BASELINE,
        "selection_gap_per_square": best_g7_persq - sel_g7_persq,
        "selection_gap_piece_only": best_g7_piece - sel_g7_piece,
        "vs_stage5_5": {"selected_minus_baseline_persq": sel_g7_persq - STAGE55_BASELINE["per_square"],
                        "best_minus_baseline_persq": best_g7_persq - STAGE55_BASELINE["per_square"]},
    }
    Path(f"{RESULTS_DIR}/selection_confound_game7.json").write_text(json.dumps(confound, indent=2))
    print("\n=== SELECTION-CONFOUND CHECK (game7 diagnostic) ===")
    print(f"  game2-SELECTED ckpt (ep{sel_epoch}): game7 per-sq={sel_g7_persq:.4f}  piece-only={sel_g7_piece:.4f}")
    print(f"  best-by-GAME7 ckpt  (ep{int(_diag['epoch'])}): game7 per-sq={best_g7_persq:.4f}  piece-only={best_g7_piece:.4f}")
    if confound["selection_gap_per_square"] > 0.01 or confound["selection_gap_piece_only"] > 0.01:
        print("\033[93m  [FLAG] best-by-game7 substantially > game2-selected.\033[0m")
    else:
        print("  [ok] game2-selection and best-by-game7 agree (no material selection confound).")
else:
    print("\n[skip] selection-confound check (per-epoch game7 diagnostic was off; no best_game7_diag.pt).")
# ==============================================================================================


# %% [Cell 16 — recipe.json (reportable) + summary]
recipe = {
    "arch": "DINOv2-ViT-S/14", "mode": MODE, "run_name": RUN_NAME, "seed": SEED,
    "param_count": int(n_total), "source_weights": SOURCE, "dino_load_path": DINO_LOAD_PATH,
    "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}", "patch_tokens": int(N_PATCHES),
    "resolution_note": (f"100x100 crops (byte-identical to ResNet/ConvNeXt) resized to {INPUT_SIZE} "
                        f"(ViT-S/14 -> {N_PATCHES} patch tokens) immediately before ImageNet-normalize, "
                        f"in both train and eval. 224 is native DINOv2 resolution."),
    "optimizer": "AdamW", "betas": [0.9, 0.999], "weight_decay": WEIGHT_DECAY,
    "lr_schedule": "cosine, eta_min=0.01*lr_head", "lr_head": LR_HEAD,
    "lr_backbone": ("frozen (linprobe)" if FREEZE_ALL else LR_BACKBONE),
    "epochs": NUM_EPOCHS, "warmup_phaseA_epochs": (None if FREEZE_ALL else WARMUP_EPOCHS),
    "early_stop_patience": EARLY_STOP_PATIENCE, "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM, "effective_batch": BATCH_SIZE * GRAD_ACCUM,
    "label_smoothing_train_ce": LABEL_SMOOTHING, "ablation": "label_smoothing vs combined_game6 (train-CE only)",
    "freeze_scheme": ("linprobe: backbone (patch_embed/blocks/norm/cls_token/pos_embed) FROZEN all "
                      "epochs, head only." if FREEZE_ALL else
                      "Phase A: freeze whole backbone, train head; Phase B: unfreeze all, discriminative "
                      "LRs. LayerNorm only (no BatchNorm) -> BN-freeze N/A."),
    "augmentation": "color-jitter only (zeroshot)" if MODE == "zeroshot"
                    else "jitter@0.7 -> shear@0.8(±8°) -> noise@0.5(std=0.015)",
    "selection_metric": SELECT_ON, "selected_epoch": int(best_select_epoch),
    "sampler": ("REAL-ONLY ablation: uniform RandomSampler over real, replacement=True, 100k draws/epoch (NO synth)"
                if ARGS.no_synth else
                "WeightedRandomSampler 50/50 synth/real, 100k draws/epoch"
                if MODE in ("stage5", "linprobe") else "shuffle (natural distribution)"),
    "no_synth_ablation": bool(ARGS.no_synth),
    "data": ("REAL-ONLY: 30 manual (games 8-11) + PGN " + ",".join(str(g) for g in TRAIN_PGN_GAMES)
             + " (NO synthetic data)") if ARGS.no_synth else
            {"zeroshot": "full dataset_v1 synth (90/10 by-image split for synth-val selection)",
             "stage3": "30 manual + game4 + game5 PGN (~323 frames real)",
             "stage5": "dataset_v1 synth + 30 manual + game4 + game5 PGN (combined)",
             "linprobe": "dataset_v1 synth + 30 manual + game4 + game5 PGN (combined, frozen backbone)"}[MODE],
    # Split metadata (defaults reproduce the original split; new mini-experiment overrides via flags).
    "split": {"train_manual_games": [8, 9, 10, 11], "train_pgn_games": TRAIN_PGN_GAMES,
              "val_game": VAL_GAME, "test_games": HELD_OUT_GAMES},
    "results": {"heldout_per_square": persq, "heldout_piece_only": piece, "heldout_empty": empty,
                "heldout_games": HELD_OUT_GAMES, "val_game": VAL_GAME, "val_per_square": rv_acc,
                "game7_per_square": rv_acc,  # kept for build_report back-compat (original runs: val==game7)
                "forgetting_delta": sm_acc - PRE_SYNTH_MONITOR_ACC},
}
Path(f"{RESULTS_DIR}/recipe.json").write_text(json.dumps(recipe, indent=2))
print(f"wrote {RESULTS_DIR}/recipe.json")
print("\033[92m✓ All cells complete — run finished.\033[0m")
