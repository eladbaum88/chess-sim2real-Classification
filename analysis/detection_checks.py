"""
Crop-distribution diagnostic: compare synth vs real crops.
Tests whether real crops contain more black-border pixels than synth.
"""
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import cv2
import csv
from pathlib import Path

# Project modules
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (
    warp_chessboard_image, crop_square,
    find_corners, ChessboardNotLocatedException,
)

# --- Configuration ------------------------------------------------------------
SYNTH_MANIFEST = "/home/eladbaum/chess_project/preprocessing/manifest.csv"
SYNTH_IMAGES_DIR = "/home/eladbaum/chess_project/data/dataset_v1/images"
REAL_GT_CSV = "/home/eladbaum/chess_project/data/game7_per_frame/gt.csv"
REAL_IMAGES_DIR = "/home/eladbaum/chess_project/data/game7_per_frame/images"
OUT_DIR = "/home/eladbaum/chess_project/training/resnet18/Real_Only/plots"
N_FRAMES_EACH = 20  # how many frames from each dataset to analyze
BLACK_THRESHOLD = 20  # pixel intensity below this counts as "black"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Helpers ------------------------------------------------------------------
def crop_with_pipeline(bgr_image):
    """Same warp+crop chain as the training datasets use. Returns list of 64 crops."""
    H, W = bgr_image.shape[:2]
    try:
        np.random.seed(42)
        corners = find_corners(bgr_image)
        # OOB check, same as in RealGameDataset
        lo, hi_x, hi_y = -8, W + 8, H + 8
        in_bounds = bool(np.all(
            (corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
            & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)
        ))
        if not in_bounds:
            raise ChessboardNotLocatedException("OOB")
        used_fallback = False
    except Exception:
        corners = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
            dtype=np.float32,
        )
        used_fallback = True

    warped = warp_chessboard_image(bgr_image, corners)
    crops = []
    for br in range(8):
        for bc in range(8):
            c_bgr = crop_square(warped, br, bc)
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            crops.append((c_rgb, br, bc))
    return crops, used_fallback


def measure_black_fraction(crop_rgb, threshold=BLACK_THRESHOLD):
    """Fraction of pixels whose all 3 channels are below threshold (i.e. black)."""
    # treat a pixel as "black" if max channel value < threshold
    is_black = (crop_rgb.max(axis=2) < threshold)
    return float(is_black.mean())


def collect_crops(image_paths, label="?"):
    """Returns list of (crop_rgb, black_fraction, source_frame, board_row, board_col, used_fallback)."""
    out = []
    n_fallback_frames = 0
    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [warn] couldn't read {img_path}")
            continue
        try:
            crops, used_fallback = crop_with_pipeline(bgr)
        except Exception as e:
            print(f"  [warn] crop pipeline failed on {img_path}: {e}")
            continue
        if used_fallback:
            n_fallback_frames += 1
        for c_rgb, br, bc in crops:
            blk = measure_black_fraction(c_rgb)
            out.append({
                "crop_rgb": c_rgb,
                "black_frac": blk,
                "source": str(img_path),
                "board_row": br,
                "board_col": bc,
                "used_fallback": used_fallback,
            })
    print(f"  [{label}] {len(image_paths)} frames, {n_fallback_frames} used corner-fallback "
          f"({n_fallback_frames/len(image_paths)*100:.0f}%)")
    return out


# --- Collect synth frames -----------------------------------------------------
print("=" * 60)
print("Collecting SYNTH crops...")
print("=" * 60)
synth_frames = []
with open(SYNTH_MANIFEST) as f:
    seen_frames = set()
    for r in csv.DictReader(f):
        # manifest has one row per square; we want unique frames
        # scripts/manifest.csv schema uses source_image (one row per square)
        img_path = Path(SYNTH_IMAGES_DIR) / r["source_image"]
        if img_path in seen_frames:
            continue
        seen_frames.add(img_path)
        if img_path.exists():
            synth_frames.append(img_path)
        if len(synth_frames) >= N_FRAMES_EACH:
            break

synth_crops = collect_crops(synth_frames, label="synth")

# --- Collect real frames ------------------------------------------------------
print()
print("=" * 60)
print("Collecting REAL (game7) crops...")
print("=" * 60)
real_frames = []
with open(REAL_GT_CSV) as f:
    for r in csv.DictReader(f):
        img_path = Path(REAL_IMAGES_DIR) / r["image_name"]
        if img_path.exists():
            real_frames.append(img_path)
        if len(real_frames) >= N_FRAMES_EACH:
            break

real_crops = collect_crops(real_frames, label="real")

# --- Numerical summary --------------------------------------------------------
synth_black = np.array([c["black_frac"] for c in synth_crops])
real_black = np.array([c["black_frac"] for c in real_crops])

print()
print("=" * 60)
print("BLACK-PIXEL FRACTION PER CROP — distribution stats")
print("=" * 60)
print(f"{'':>12s}  {'synth':>10s}  {'real':>10s}")
print(f"{'n_crops':>12s}  {len(synth_black):>10d}  {len(real_black):>10d}")
print(f"{'mean':>12s}  {synth_black.mean():>10.4f}  {real_black.mean():>10.4f}")
print(f"{'median':>12s}  {np.median(synth_black):>10.4f}  {np.median(real_black):>10.4f}")
print(f"{'p90':>12s}  {np.percentile(synth_black, 90):>10.4f}  {np.percentile(real_black, 90):>10.4f}")
print(f"{'p99':>12s}  {np.percentile(synth_black, 99):>10.4f}  {np.percentile(real_black, 99):>10.4f}")
print(f"{'max':>12s}  {synth_black.max():>10.4f}  {real_black.max():>10.4f}")

print()
print("Crops with >5% black pixels:")
print(f"  synth:  {(synth_black > 0.05).sum():>5d}/{len(synth_black)}  ({(synth_black > 0.05).mean()*100:.1f}%)")
print(f"  real:   {(real_black > 0.05).sum():>5d}/{len(real_black)}  ({(real_black > 0.05).mean()*100:.1f}%)")

print()
print("Crops with >20% black pixels (heavy warp artifact):")
print(f"  synth:  {(synth_black > 0.20).sum():>5d}/{len(synth_black)}  ({(synth_black > 0.20).mean()*100:.1f}%)")
print(f"  real:   {(real_black > 0.20).sum():>5d}/{len(real_black)}  ({(real_black > 0.20).mean()*100:.1f}%)")


# --- Visualization: histograms ------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 41)
ax.hist(synth_black, bins=bins, alpha=0.5, label=f"synth ({len(synth_black)} crops)", color="C0")
ax.hist(real_black, bins=bins, alpha=0.5, label=f"real game7 ({len(real_black)} crops)", color="C1")
ax.set_xlabel("fraction of black pixels per crop")
ax.set_ylabel("# crops")
ax.set_title(f"Black-pixel fraction distribution — {N_FRAMES_EACH} frames each\n"
             f"(threshold: pixel is 'black' if max channel < {BLACK_THRESHOLD})")
ax.legend()
ax.set_yscale("log")  # log scale because most crops will cluster near 0
hist_path = f"{OUT_DIR}/synth_vs_real_black_fraction.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=120)
plt.close()
print(f"\nsaved histogram: {hist_path}")


# --- Visualization: sample crops from each, sorted by black fraction ---------
def make_sample_grid(crops, title, out_path, n=8):
    """Show n crops sorted from least to most black, evenly spaced."""
    sorted_crops = sorted(crops, key=lambda c: c["black_frac"])
    if len(sorted_crops) < n:
        n = len(sorted_crops)
    # pick n evenly-spaced positions in the sorted list
    indices = np.linspace(0, len(sorted_crops) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for ax, idx in zip(axes, indices):
        c = sorted_crops[idx]
        ax.imshow(c["crop_rgb"])
        ax.set_title(f"black={c['black_frac']*100:.0f}%", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved grid: {out_path}")

make_sample_grid(
    synth_crops,
    "SYNTH crops — sorted by black-pixel fraction (low → high)",
    f"{OUT_DIR}/synth_crop_samples.png",
)
make_sample_grid(
    real_crops,
    "REAL game7 crops — sorted by black-pixel fraction (low → high)",
    f"{OUT_DIR}/real_crop_samples.png",
)

print()
print("=" * 60)
print("DONE — review the three saved images:")
print(f"  - {hist_path}")
print(f"  - {OUT_DIR}/synth_crop_samples.png")
print(f"  - {OUT_DIR}/real_crop_samples.png")
print("=" * 60)