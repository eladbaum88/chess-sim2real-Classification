"""Stratified sanity check for dataset_v1.

Picks 10 samples (3 sparse + 3 medium + 3 dense + 1 random by piece count;
≥4 distinct HDRIs) and verifies that the rendered pixels match the FEN
labels.

Outputs:
  dataset_v1/sanity/sample_<i>__<image>.png   per-sample side-by-side:
      left   = rendered image with grid + labels (yellow=white, magenta=black)
      right  = rendered image with grid only (no labels)
  dataset_v1/sanity/diagnostic_K.png          lone-K-on-d5 in all 3 cameras
                                              (one row per camera, label
                                              overlaid; visually verify the
                                              K appears in the correct
                                              labeled square in each)
  dataset_v1/sanity/samples_contact.png       contact sheet of all 10

Console output includes a 1-line FEN-vs-label parity check and a printed
8x8 grid for 2 of the 10 samples.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_DIR / "dataset_v1"
CSV = DATASET_DIR / "labels.csv"
IMAGES_DIR = DATASET_DIR / "images"
SANITY_DIR = DATASET_DIR / "sanity"

# Diagnostic FEN: lone white king on d5 (an asymmetric square, not on the
# center or a diagonal of symmetry). If the K appears at the correct grid
# cell in all 3 cameras after applying the per-camera transform, the
# transform is correct.
DIAGNOSTIC_FEN = "8/8/8/3K4/8/8/8/8"
DIAGNOSTIC_TAG = "K_on_d5"

# Per-camera FEN-grid transform. v1's render script orders rectified corners
# by image position (order_corners_tl_tr_br_bl), and the camera ends up
# 180° from white-POV for all 3 cameras when --view=black is used.
VIEW_TRANSFORMS = {
    "1_overhead": "rot180",
    "2_west":     "rot180",
    "3_east":     "rot180",
}

PIECE_TO_CLASS = {"P":0,"R":1,"N":2,"B":3,"Q":4,"K":5,"p":6,"r":7,"n":8,"b":9,"q":10,"k":11}
CLASS_TO_PIECE = {v:k for k,v in PIECE_TO_CLASS.items()}; CLASS_TO_PIECE[12] = "."
EMPTY_CLASS = 12

UNICODE_PIECES = {
    "K":"♔","Q":"♕","R":"♖","B":"♗","N":"♘","P":"♙",
    "k":"♚","q":"♛","r":"♜","b":"♝","n":"♞","p":"♟",
}

# Short tokens for terminal grid dump.
SHORT = {
    "K":"wK","Q":"wQ","R":"wR","B":"wB","N":"wN","P":"wP",
    "k":"bK","q":"bQ","r":"bR","b":"bB","n":"bN","p":"bP",
    ".":". ",
}


def fen_piece_count(fen):
    return sum(1 for c in fen.split()[0] if c.isalpha())


def fen_to_grid_white_pov(fen):
    """FEN -> 8x8 char grid. grid[0]=rank 8 (top), grid[7]=rank 1 (bottom).
    This is the canonical white-POV orientation, NOT yet transformed for
    the camera view."""
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
            raise ValueError(f"Rank {r} of {fen!r} doesn't sum to 8")
    return np.array(grid)


def apply_transform(grid, name):
    if name == "identity":      return grid
    if name == "rot90":         return np.rot90(grid, 1)
    if name == "rot180":        return np.rot90(grid, 2)
    if name == "rot270":        return np.rot90(grid, 3)
    if name == "fliplr":        return np.fliplr(grid)
    if name == "flipud":        return np.flipud(grid)
    raise ValueError(f"Unknown transform: {name}")


def grid_to_class_tensor(char_grid):
    """char_grid (np.ndarray of strings) -> int 8x8 class tensor."""
    out = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)
    for r in range(8):
        for c in range(8):
            ch = char_grid[r, c]
            if ch in PIECE_TO_CLASS:
                out[r, c] = PIECE_TO_CLASS[ch]
    return out


def print_grid(char_grid, title):
    """Dump 8x8 grid to terminal using wP/bR etc tokens."""
    print(f"\n{title}")
    print("    " + "   ".join("a b c d e f g h".split()))
    for r in range(8):
        rank_num = 8 - r
        row = "  ".join(SHORT[char_grid[r, c]] for c in range(8))
        print(f"{rank_num}   {row}")


# ----------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------
def stratify_samples(df, seed=2026, target=10):
    """Pick `target` samples stratified by piece count.
    Layout: target // 3 per bucket (sparse/medium/dense), remainder as random.
    Ensures ≥4 distinct HDRIs when feasible. Falls back gracefully if the
    pool is too small."""
    df = df.copy()
    df["piece_count"] = df["fen"].apply(fen_piece_count)
    df["bucket"] = pd.cut(
        df["piece_count"], bins=[-1, 12, 24, 32], labels=["sparse", "medium", "dense"]
    )

    per_bucket = max(1, target // 3)
    n_random = max(0, target - 3 * per_bucket)
    rng = random.Random(seed)
    bucket_counts = {b: len(df[df["bucket"] == b]) for b in ["sparse", "medium", "dense"]}
    if min(bucket_counts.values()) < per_bucket:
        print(f"NOTE: pool too small for stratification "
              f"(bucket counts = {bucket_counts}, need {per_bucket}/bucket); "
              f"sampling {min(target, len(df))} uniformly for smoke check.")
        chosen_idx = rng.sample(list(df.index), min(target, len(df)))
        return df.loc[chosen_idx].reset_index(drop=True)

    chosen_idx = []
    for bucket in ["sparse", "medium", "dense"]:
        sub = df[df["bucket"] == bucket]
        # Diversify by HDRI within the bucket if possible
        chosen = []
        hdris_seen = set()
        candidates = list(sub.index)
        rng.shuffle(candidates)
        for idx in candidates:
            if len(chosen) >= per_bucket:
                break
            h = df.loc[idx, "hdri"]
            if h not in hdris_seen or len(chosen) >= max(1, per_bucket - 1):
                chosen.append(idx)
                hdris_seen.add(h)
        if len(chosen) < per_bucket:
            chosen = candidates[:per_bucket]
        chosen_idx.extend(chosen)

    # Random picks for whatever's left of `target`
    remaining = [i for i in df.index if i not in chosen_idx]
    rng.shuffle(remaining)
    chosen_idx.extend(remaining[:n_random])

    # HDRI diversity check (≥4 distinct)
    chosen_df = df.loc[chosen_idx].copy()
    distinct_hdris = chosen_df["hdri"].nunique()
    if distinct_hdris < 4:
        # Try to substitute one same-bucket sample for HDRI diversity
        underrepresented = set(df["hdri"].unique()) - set(chosen_df["hdri"])
        for h in underrepresented:
            if distinct_hdris >= 4:
                break
            alt_rows = df[df["hdri"] == h]
            if alt_rows.empty:
                continue
            alt_idx = rng.choice(alt_rows.index.tolist())
            # Swap with the LAST element of the most-represented HDRI in chosen
            most_common = chosen_df["hdri"].value_counts().idxmax()
            swappable = chosen_df[chosen_df["hdri"] == most_common].index
            if len(swappable) > 1:
                drop_idx = swappable[-1]
                chosen_df = chosen_df.drop(drop_idx)
                chosen_df = pd.concat([chosen_df, df.loc[[alt_idx]]])
                distinct_hdris = chosen_df["hdri"].nunique()

    if chosen_df["hdri"].nunique() < 4:
        print(f"WARN: only {chosen_df['hdri'].nunique()} distinct HDRIs in "
              f"the dataset; can't reach the 4-HDRI floor.")
    return chosen_df.sort_values(["bucket", "piece_count"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_image_with_labels(ax, img, label_char_grid, title):
    ax.imshow(img)
    W, H = img.size
    sw, sh = W / 8, H / 8
    for i in range(9):
        ax.axhline(i * sh, color="cyan", linewidth=0.6, alpha=0.6)
        ax.axvline(i * sw, color="cyan", linewidth=0.6, alpha=0.6)
    for r in range(8):
        for c in range(8):
            ch = label_char_grid[r, c]
            if ch == ".":
                continue
            color = "yellow" if ch.isupper() else "magenta"
            ax.text(c * sw + sw/2, r * sh + sh/2, ch,
                    color=color, fontsize=12, ha="center", va="center",
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def draw_image_grid_only(ax, img, title):
    ax.imshow(img)
    W, H = img.size
    sw, sh = W / 8, H / 8
    for i in range(9):
        ax.axhline(i * sh, color="cyan", linewidth=0.6, alpha=0.6)
        ax.axvline(i * sw, color="cyan", linewidth=0.6, alpha=0.6)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ----------------------------------------------------------------------
# Diagnostic render of an asymmetric FEN
# ----------------------------------------------------------------------
def ensure_diagnostic_renders():
    """Render the lone-K FEN if its 3 images aren't already present in
    dataset_v1/images. Returns paths in (1_overhead, 2_west, 3_east) order."""
    expected = {cam: IMAGES_DIR / f"fen_diag_{DIAGNOSTIC_TAG}_{cam}.png"
                for cam in ["1_overhead", "2_west", "3_east"]}
    missing = [c for c, p in expected.items() if not p.exists()]
    if missing:
        print(f"Diagnostic images missing for {missing} — invoking build_dataset_v1 "
              f"--custom-fen on the cluster GPU...")
        cmd = [
            "srun", "--partition=gpu", "--gres=gpu:1",
            "--time=00:05:00", "--job-name=ds_v1_diag",
            "bash", "-c",
            f"cd {PROJECT_DIR} && python build_dataset_v1.py "
            f"--custom-fen '{DIAGNOSTIC_FEN}' --custom-tag {DIAGNOSTIC_TAG} "
            f"--outer-padding 0.03 --piece-margin 0.2"
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"Diagnostic render failed (rc={rc})")
    return expected


def write_diagnostic_overlay(image_paths):
    fen_grid = fen_to_grid_white_pov(DIAGNOSTIC_FEN)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (cam, path) in zip(axes, image_paths.items()):
        img = Image.open(path)
        xform = VIEW_TRANSFORMS[cam]
        grid_aligned = apply_transform(fen_grid, xform)
        draw_image_with_labels(
            ax, img, grid_aligned,
            f"{cam}  xform={xform}  K should land on its label\n"
            f"FEN: {DIAGNOSTIC_FEN}"
        )
    fig.suptitle(
        "Diagnostic: lone white K on d5 — confirm K visually sits in the labeled square",
        fontsize=11, y=1.02)
    plt.tight_layout()
    out = SANITY_DIR / "diagnostic_K.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote diagnostic overlay: {out}")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--target", type=int, default=10,
                   help="Number of stratified samples (target//3 per bucket).")
    p.add_argument("--skip-diagnostic-render", action="store_true",
                   help="Don't auto-render the diagnostic FEN if missing.")
    args = p.parse_args()

    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV}")
    print(f"Piece-count distribution in pool:\n"
          f"  sparse (≤12): {(df['fen'].apply(fen_piece_count) <= 12).sum()}\n"
          f"  medium (13-24): {((df['fen'].apply(fen_piece_count) >= 13) & (df['fen'].apply(fen_piece_count) <= 24)).sum()}\n"
          f"  dense  (≥25): {(df['fen'].apply(fen_piece_count) >= 25).sum()}\n"
          f"  HDRIs: {df['hdri'].nunique()} distinct")

    chosen = stratify_samples(df, seed=args.seed, target=args.target)
    print(f"\nSelected {len(chosen)} samples:")
    for _, row in chosen.iterrows():
        n = fen_piece_count(row["fen"])
        print(f"  fen_{int(row['fen_idx']):04d} | {row['camera']:11s} | "
              f"pieces={n:2d} ({row['bucket']}) | hdri={row['hdri']}")
    distinct_hdris = chosen["hdri"].nunique()
    print(f"\nDistinct HDRIs in selection: {distinct_hdris}")

    # Per-sample side-by-side panels
    per_sample_paths = []
    sample_class_grids = []  # for FEN-vs-label parity check
    rng = random.Random(args.seed)
    grid_dump_indices = sorted(rng.sample(range(len(chosen)), 2))

    for i, (_, row) in enumerate(chosen.iterrows()):
        img_path = IMAGES_DIR / Path(row["image_path"]).name
        img = Image.open(img_path)
        cam = row["camera"]
        xform = VIEW_TRANSFORMS.get(cam, "identity")
        char_grid = apply_transform(fen_to_grid_white_pov(row["fen"]), xform)
        class_grid = grid_to_class_tensor(char_grid)
        sample_class_grids.append(class_grid)

        # Per-image figure
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        n = fen_piece_count(row["fen"])
        title_l = (f"sample {i}: {Path(row['image_path']).name}\n"
                   f"cam={cam}  xform={xform}  pieces={n} ({row['bucket']})\n"
                   f"FEN: {row['fen'].split()[0]}\nHDRI: {row['hdri']}")
        draw_image_with_labels(ax[0], img, char_grid, title_l)
        draw_image_grid_only(ax[1], img, "(b) grid only — verify pieces look right")
        plt.tight_layout()
        out = SANITY_DIR / f"sample_{i:02d}__{Path(row['image_path']).stem}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        per_sample_paths.append(out)

        # Dump 8x8 grid for the 2 chosen samples
        if i in grid_dump_indices:
            print_grid(
                char_grid,
                f"Sample {i} grid dump (fen_{int(row['fen_idx']):04d} {cam}, "
                f"xform={xform}, FEN={row['fen'].split()[0]})"
            )

    # Contact sheet (2 cols per sample: with-labels | grid-only)
    cols = 4  # 2 samples per row in the contact sheet
    rows = (len(chosen) + cols//2 - 1) // (cols//2)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for i, (_, row) in enumerate(chosen.iterrows()):
        img = Image.open(IMAGES_DIR / Path(row["image_path"]).name)
        cam = row["camera"]; xform = VIEW_TRANSFORMS.get(cam, "identity")
        char_grid = apply_transform(fen_to_grid_white_pov(row["fen"]), xform)
        r = i // (cols // 2)
        c0 = (i % (cols // 2)) * 2
        draw_image_with_labels(
            axes[r, c0], img, char_grid,
            f"#{i} {Path(row['image_path']).name}\n{cam} pieces={fen_piece_count(row['fen'])}"
        )
        draw_image_grid_only(axes[r, c0 + 1], img, f"#{i} (grid only)")
    # hide unused
    for k in range(len(chosen) * 2, rows * cols):
        axes[k // cols, k % cols].axis("off")
    plt.suptitle("dataset_v1 stratified sanity samples — left: image+labels, right: grid only",
                 fontsize=10, y=1.0)
    plt.tight_layout()
    contact = SANITY_DIR / "samples_contact.png"
    plt.savefig(contact, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote per-sample images: {len(per_sample_paths)} into {SANITY_DIR}")
    print(f"Wrote contact sheet: {contact}")

    # Diagnostic: lone-K FEN
    if not args.skip_diagnostic_render:
        diag_paths = ensure_diagnostic_renders()
        write_diagnostic_overlay(diag_paths)

    # FEN-vs-label parity check (the 1-line summary)
    all_grids = np.stack(sample_class_grids, axis=0)
    label_empty_count = int((all_grids == EMPTY_CLASS).sum())
    fen_empty_count = 0
    for _, row in chosen.iterrows():
        fen_empty_count += 64 - fen_piece_count(row["fen"])
    parity_ok = (label_empty_count == fen_empty_count)
    parity_msg = (f"PARITY: FEN empties = {fen_empty_count}, label==12 count = "
                  f"{label_empty_count} → {'OK' if parity_ok else 'BROKEN'}")
    print("\n" + "=" * 60)
    print(parity_msg)
    print("=" * 60)
    if not parity_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
