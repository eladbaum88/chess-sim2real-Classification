"""
build_correct_vs_wrong_plots.py — for both the zero_shot and Real_Only
checkpoints, run inference over game7 and save one PNG per model showing
example crops that were classified correctly vs incorrectly.

Layout per PNG (same for both models):
  Top half:    4×8 grid of crops the model got RIGHT, stratified across classes.
  Bottom half: 4×8 grid of crops the model got WRONG, stratified across the
               top-confusion pairs first, then random.
Annotation under each crop:
  correct:  "<true label>"          (green)
  wrong:    "<true> → <pred>"        (red)

Outputs:
  zero_shot/plots/predictions_correct_vs_wrong.png
  Real_Only/plots/predictions_correct_vs_wrong.png
"""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/preprocessing")
from verify_woelflein_crops import (
    warp_chessboard_image, crop_square,
    find_corners, ChessboardNotLocatedException,
)
from fen_to_grid import fen_to_label_grid


PROJECT_ROOT = Path("/home/eladbaum/chess_project")
GAME7_GT_CSV = PROJECT_ROOT / "data/game7_per_frame/gt.csv"
GAME7_IMAGES = PROJECT_ROOT / "data/game7_per_frame/images"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

NUM_CLASSES = 13
CLASS_NAMES = [
    "White Pawn",   "White Rook",   "White Knight", "White Bishop",
    "White Queen",  "White King",
    "Black Pawn",   "Black Rook",   "Black Knight", "Black Bishop",
    "Black Queen",  "Black King",
    "Empty",
]
CLASS_SHORT = ["WP", "WR", "WN", "WB", "WQ", "WK",
               "BP", "BR", "BN", "BB", "BQ", "BK", "."]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)

GRID_ROWS, GRID_COLS = 4, 8   # 32 samples per panel
CROP_DISPLAY = 96             # px per crop in the final plot


# --------------------------------------------------------------------------
class Game7InferenceDataset(Dataset):
    """Game7 per-square dataset. Same find_corners-with-OOB-fallback path the
    training script uses. Returns (tensor, label, manifest_idx). The manifest
    is exposed so the visualization can look up (image, row, col) by index."""

    CORNER_OOB_TOLERANCE = 8

    def __init__(self, gt_csv_path, images_dir):
        self.images_dir = Path(images_dir)
        rows = []
        with open(gt_csv_path) as f:
            for r in csv.DictReader(f):
                grid = fen_to_label_grid(r["fen"], "game7")
                for br in range(8):
                    for bc in range(8):
                        rows.append({
                            "image_name": r["image_name"],
                            "board_row": br,
                            "board_col": bc,
                            "label": int(grid[br, bc]),
                            "fen": r["fen"],
                        })
        import pandas as pd
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
            c = find_corners(bgr)
            lo = -self.CORNER_OOB_TOLERANCE
            in_bounds = bool(np.all(
                (c[:, 0] >= lo) & (c[:, 0] <= W + self.CORNER_OOB_TOLERANCE)
                & (c[:, 1] >= lo) & (c[:, 1] <= H + self.CORNER_OOB_TOLERANCE)
            ))
            if not in_bounds:
                raise ChessboardNotLocatedException("OOB")
        except Exception:
            c = np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                dtype=np.float32,
            )
        self._corner_cache[image_name] = c
        return c

    def get_crop_rgb(self, idx):
        """Return the HWC uint8 RGB crop for sample idx (no tensorization)."""
        row = self.manifest.iloc[idx]
        bgr = cv2.imread(str(self.images_dir / row["image_name"]))
        corners = self._get_corners(row["image_name"], bgr)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, int(row["board_row"]), int(row["board_col"]))
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        crop_rgb = self.get_crop_rgb(idx)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(crop_rgb))
                 .permute(2, 0, 1).float() / 255.0
        )
        return tensor, int(self.manifest.iloc[idx]["label"]), idx


def build_model():
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)


@torch.no_grad()
def run_inference(model, dataset):
    """Return arrays of shape (N,): preds, labels, idxs (in dataset order)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True)
    all_preds, all_labels, all_idxs = [], [], []
    for xb, yb, ib in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        xb = (xb - IMAGENET_MEAN) / IMAGENET_STD
        logits = model(xb)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(yb.numpy())
        all_idxs.append(ib.numpy())
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_idxs))


def stratified_sample(indices_per_class, total, rng):
    """Try to give every class at least one slot; fill the rest by round-robin."""
    classes = [c for c in indices_per_class if indices_per_class[c]]
    if not classes:
        return []
    picks = []
    seen = {c: 0 for c in classes}
    while len(picks) < total:
        progress = False
        for c in classes:
            if len(picks) >= total:
                break
            if seen[c] < len(indices_per_class[c]):
                idx = indices_per_class[c][seen[c]]
                picks.append(idx)
                seen[c] += 1
                progress = True
        if not progress:
            break
    return picks


def pick_correct_samples(preds, labels, n, rng):
    """Pick n correctly-classified samples stratified across true classes."""
    correct = np.where(preds == labels)[0]
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for i in correct:
        by_class[int(labels[i])].append(int(i))
    for c in by_class:
        rng.shuffle(by_class[c])
    return stratified_sample(by_class, n, rng)


def pick_wrong_samples(preds, labels, n, rng):
    """Pick n wrong samples. Spread across (true, pred) confusion pairs,
    prioritizing the most frequent pairs (so the user sees common errors first)."""
    wrong = np.where(preds != labels)[0]
    by_pair = {}
    for i in wrong:
        key = (int(labels[i]), int(preds[i]))
        by_pair.setdefault(key, []).append(int(i))
    # Sort pairs by frequency descending — common errors first
    pair_order = sorted(by_pair.keys(), key=lambda k: -len(by_pair[k]))
    for k in by_pair:
        rng.shuffle(by_pair[k])
    seen = {k: 0 for k in by_pair}
    picks = []
    while len(picks) < n:
        progress = False
        for k in pair_order:
            if len(picks) >= n:
                break
            if seen[k] < len(by_pair[k]):
                picks.append(by_pair[k][seen[k]])
                seen[k] += 1
                progress = True
        if not progress:
            break
    return picks


def make_plot(dataset, preds, labels, model_label, accuracy, out_path):
    rng = random.Random(SEED)
    n_total = GRID_ROWS * GRID_COLS

    correct_idxs = pick_correct_samples(preds, labels, n_total, rng)
    wrong_idxs = pick_wrong_samples(preds, labels, n_total, rng)

    print(f"  correct examples: {len(correct_idxs)}/{n_total}")
    print(f"  wrong examples:   {len(wrong_idxs)}/{n_total}")

    # Build figure: two large panels stacked
    fig = plt.figure(figsize=(GRID_COLS * 1.6, GRID_ROWS * 2 * 1.8 + 1.6))
    gs = fig.add_gridspec(
        nrows=2 * GRID_ROWS + 2, ncols=GRID_COLS,
        height_ratios=[0.4] + [1] * GRID_ROWS + [0.4] + [1] * GRID_ROWS,
        hspace=0.45, wspace=0.15,
    )

    # Top banner
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    ax_top.text(0.5, 0.45,
                f"{model_label}  —  game7 per-square accuracy: {accuracy:.4f}",
                ha="center", va="center", fontsize=16, fontweight="bold")
    ax_top.text(0.5, -0.05,
                "Top: correctly classified.   Bottom: misclassified  (true → predicted).",
                ha="center", va="center", fontsize=11, color=(0.3, 0.3, 0.3))

    # Helper to draw a grid panel
    def draw_panel(start_row, idxs, kind):
        for slot in range(n_total):
            r = start_row + slot // GRID_COLS
            c = slot % GRID_COLS
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            if slot >= len(idxs):
                ax.axis("off")
                continue
            mi = idxs[slot]
            crop = dataset.get_crop_rgb(mi)
            ax.imshow(crop)
            true_lbl = int(labels[mi])
            pred_lbl = int(preds[mi])
            if kind == "correct":
                title = CLASS_SHORT[true_lbl]
                color = (0.05, 0.55, 0.10)
            else:
                title = f"{CLASS_SHORT[true_lbl]}→{CLASS_SHORT[pred_lbl]}"
                color = (0.75, 0.05, 0.05)
            ax.set_title(title, fontsize=9, color=color, pad=2)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    # Section headers
    ax_sec1 = fig.add_subplot(gs[1 + GRID_ROWS, :])
    ax_sec1.axis("off")
    ax_sec1.text(0.5, 0.5, "MISCLASSIFIED  (true → predicted)",
                 ha="center", va="center", fontsize=13, fontweight="bold",
                 color=(0.75, 0.05, 0.05))

    draw_panel(1, correct_idxs, "correct")
    draw_panel(2 + GRID_ROWS, wrong_idxs, "wrong")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading game7 dataset ...")
    ds = Game7InferenceDataset(GAME7_GT_CSV, GAME7_IMAGES)
    print(f"  {len(ds)} samples, {ds.manifest['image_name'].nunique()} frames")

    models_to_run = [
        ("Zero-shot (synth only)",
         PROJECT_ROOT / "zero_shot/results/best_synth.pt",
         PROJECT_ROOT / "zero_shot/plots/predictions_correct_vs_wrong.png"),
        ("Real-only (games 2/4/5/6)",
         PROJECT_ROOT / "Real_Only/results/best_real.pt",
         PROJECT_ROOT / "Real_Only/plots/predictions_correct_vs_wrong.png"),
    ]

    for label, ckpt_path, out_path in models_to_run:
        print(f"\n=== {label} ===")
        print(f"  ckpt: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model()
        model.load_state_dict(ckpt["model_state_dict"])
        preds, labels, idxs = run_inference(model, ds)
        # idxs are in dataset order since shuffle=False; preds[i] aligns to ds[idxs[i]]
        # Rearrange to manifest order (which is the same since shuffle=False, but be explicit)
        order = np.argsort(idxs)
        preds = preds[order]; labels = labels[order]
        acc = (preds == labels).mean()
        print(f"  per-square acc: {acc:.4f}")
        make_plot(ds, preds, labels, label, acc, out_path)


if __name__ == "__main__":
    main()
