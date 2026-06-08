"""Sanity check for dataset_v1: verify every image's pixels match its FEN label.

For each image we render two panels side by side:
  left  — the actual rendered image, with an 8x8 grid overlay and per-square
          piece-letter labels derived from the FEN (white = yellow, black = magenta)
  right — a clean 8x8 chessboard rendered directly from the FEN, for reference

If the letters in the overlay align with the visible pieces in the image, the
FEN matches the image. A mismatch shows up immediately (letter on the wrong
square, or a labeled square that's actually empty).

The v1 render script orders rectified corners by IMAGE position (see
order_corners_tl_tr_br_bl in chess_position_api_v2.py), so the per-camera
transform isn't the same as v2's. We probe each candidate and the user can
inspect which one aligns.

Outputs:
  dataset_v1/sanity/<image_name>__overlay.png   one per image
  dataset_v1/sanity/summary.png                 contact-sheet of all 12
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_DIR / "dataset_v1"
CSV = DATASET_DIR / "labels.csv"
IMAGES = DATASET_DIR / "images"
SANITY_DIR = DATASET_DIR / "sanity"

UNICODE_PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def fen_to_grid(fen):
    """FEN piece-placement -> 8x8 string grid. grid[0] = rank 8 (top), grid[7] = rank 1."""
    grid = [["." for _ in range(8)] for _ in range(8)]
    rows = fen.split()[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"Bad FEN: {fen}")
    for r, rank in enumerate(rows):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                grid[r][c] = ch
                c += 1
        if c != 8:
            raise ValueError(f"Rank {r} doesn't sum to 8: {rank}")
    return np.array(grid)


def apply_transform(grid, name):
    """Map a white-POV FEN grid to the image grid for a given v1 camera view.

    v1 was rendered with view='black' (black on the near side from the user's
    perspective). The v1 rectifier orders corners purely by image position
    (order_corners_tl_tr_br_bl), so each camera position lands the board on
    the image differently. These transforms were determined empirically.
    """
    if name == "identity":
        return grid
    if name == "rot90":
        return np.rot90(grid, 1)
    if name == "rot180":
        return np.rot90(grid, 2)
    if name == "rot270":
        return np.rot90(grid, 3)
    if name == "fliplr":
        return np.fliplr(grid)
    if name == "flipud":
        return np.flipud(grid)
    if name == "fliplr_rot90":
        return np.rot90(np.fliplr(grid), 1)
    if name == "fliplr_rot180":
        return np.rot90(np.fliplr(grid), 2)
    if name == "fliplr_rot270":
        return np.rot90(np.fliplr(grid), 3)
    raise ValueError(f"Unknown transform: {name}")


def render_image_with_overlay(ax, img, grid_image_aligned, title):
    """Plot rendered image with grid lines + per-square FEN letters overlaid."""
    ax.imshow(img)
    W, H = img.size
    sw, sh = W / 8, H / 8
    for i in range(9):
        ax.axhline(i * sh, color="cyan", linewidth=0.6, alpha=0.6)
        ax.axvline(i * sw, color="cyan", linewidth=0.6, alpha=0.6)
    for r in range(8):
        for c in range(8):
            ch = grid_image_aligned[r, c]
            if ch == ".":
                continue
            color = "yellow" if ch.isupper() else "magenta"
            ax.text(c * sw + sw / 2, r * sh + sh / 2, ch,
                    color=color, fontsize=11, ha="center", va="center",
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_label_only(ax, grid_image_aligned, title):
    """Clean 8x8 board rendered from the (already-transformed) label grid."""
    for r in range(8):
        for c in range(8):
            light = (r + c) % 2 == 0
            color = "#f0d9b5" if light else "#b58863"
            ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, facecolor=color, edgecolor="none"))
            ch = grid_image_aligned[r, c]
            if ch == ".":
                continue
            glyph = UNICODE_PIECES.get(ch, "?")
            pcolor = "white" if ch.isupper() else "black"
            ax.text(c + 0.5, 7 - r + 0.5, glyph, fontsize=18, ha="center", va="center",
                    color=pcolor, family="DejaVu Sans")
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transforms", default="1_overhead=rot180,2_west=rot90,3_east=rot270",
                   help="Per-camera transforms, e.g. '1_overhead=rot180,2_west=fliplr'. "
                        "Available: identity, rot90, rot180, rot270, fliplr, flipud, "
                        "fliplr_rot90, fliplr_rot180, fliplr_rot270")
    args = p.parse_args()

    transforms = {}
    for pair in args.transforms.split(","):
        k, v = pair.split("=")
        transforms[k.strip()] = v.strip()
    print(f"Transforms: {transforms}")

    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV}")

    n = len(df)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4.5, rows * 9))
    # If rows*2 == 2, axes might be (2, cols); if rows==1, also OK. Normalize:
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)

    for i, (_, row) in enumerate(df.iterrows()):
        img_path = IMAGES / Path(row["image_path"]).name
        img = Image.open(img_path)
        cam = row["camera"]
        if cam not in transforms:
            print(f"WARN: no transform for camera={cam}, using identity")
            tname = "identity"
        else:
            tname = transforms[cam]
        grid = fen_to_grid(row["fen"])
        grid_aligned = apply_transform(grid, tname)

        # Per-image individual figure
        fig_i, ax_i = plt.subplots(1, 2, figsize=(10, 5))
        render_image_with_overlay(ax_i[0], img, grid_aligned,
                                  f"{Path(row['image_path']).name} | cam={cam} | xform={tname}\n"
                                  f"FEN: {row['fen'].split()[0]}\nHDRI: {row['hdri']}")
        render_label_only(ax_i[1], grid_aligned, "expected (FEN -> image grid)")
        plt.tight_layout()
        out_path = SANITY_DIR / f"{Path(row['image_path']).stem}__overlay.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig_i)

        # Add to contact sheet
        ax_img = axes[(i // cols) * 2, i % cols]
        ax_lbl = axes[(i // cols) * 2 + 1, i % cols]
        render_image_with_overlay(ax_img, img, grid_aligned,
                                  f"{Path(row['image_path']).name} ({cam}, {tname})")
        render_label_only(ax_lbl, grid_aligned, f"FEN: {row['fen'].split()[0][:32]}...")

    # Hide any unused axes
    used = n
    for k in range(used, rows * cols):
        axes[(k // cols) * 2, k % cols].axis("off")
        axes[(k // cols) * 2 + 1, k % cols].axis("off")

    summary_path = SANITY_DIR / "summary.png"
    fig.suptitle("dataset_v1 sanity check — image (with grid+labels) above "
                 "expected FEN-derived board (below)", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote contact sheet: {summary_path}")
    print(f"Per-image overlays in: {SANITY_DIR}")


if __name__ == "__main__":
    main()
