"""
build_manifest.py — produce manifest.csv with one row per (image, board square)
giving the integer class label of that square.

FEN source of truth: Project2_3/dataset_v1/labels.csv  (one row per image).
Columns used:
    image_path  →  "images/fen_XXXX_rY_Z_<view>.png"
    camera      →  "1_overhead" / "2_west" / "3_east"
    fen         →  piece-placement string (e.g. "rnbqkbnr/pppppppp/8/...")

Discrepancy note:
  dataset_v1/images/ contains 6135 PNGs; labels.csv has 6132 rows. The 3
  extras are `fen_diag_*` diagnostic renders without FEN entries. They are
  EXCLUDED from the manifest. So the expected row count is 6132 × 64 =
  392,448 (not the 6135 × 64 = 392,640 the brief estimated).

Schema (6 columns):
    source_image, view, board_row, board_col, label, fen

Sanity output: total row count, per-class distribution, per-view balance,
plus warnings if anything looks off (empty fraction outside [40%, 70%],
missing kings, etc.).
"""

import csv
from collections import Counter
from pathlib import Path

from scripts.fen_to_grid import fen_to_label_grid


DATASET_DIR = Path("/home/eladbaum/chess_project/data_generation/dataset_v1/images")
LABELS_CSV = Path("/home/eladbaum/chess_project/data_generation/dataset_v1/labels.csv")
OUT_CSV = Path("/home/eladbaum/chess_project/manifest.csv")
VALID_VIEWS = ("overhead", "west", "east")

# Class encoding (project spec) — for the distribution report only.
CLASS_NAMES = {
    0: "P (white pawn)",
    1: "R (white rook)",
    2: "N (white knight)",
    3: "B (white bishop)",
    4: "Q (white queen)",
    5: "K (white king)",
    6: "p (black pawn)",
    7: "r (black rook)",
    8: "n (black knight)",
    9: "b (black bishop)",
    10: "q (black queen)",
    11: "k (black king)",
    12: ".  (empty)",
}


def view_from_camera_field(camera):
    """camera = '1_overhead' / '2_west' / '3_east'  →  'overhead' / 'west' / 'east'."""
    return camera.split("_", 1)[1] if "_" in camera else camera


def main():
    # ----- Read labels.csv -----
    print(f"Reading {LABELS_CSV} ...")
    with LABELS_CSV.open() as f:
        label_rows = list(csv.DictReader(f))
    print(f"  {len(label_rows)} label entries")

    # Sanity: count images on disk vs labels
    image_files = {p.name for p in DATASET_DIR.glob("*.png")}
    label_basenames = {Path(r["image_path"]).name for r in label_rows}
    extras = image_files - label_basenames
    missing = label_basenames - image_files
    print(f"  images on disk:           {len(image_files)}")
    print(f"  images in labels.csv:     {len(label_basenames)}")
    print(f"  on disk but not labelled: {len(extras)}"
          + (f"  (e.g. {sorted(extras)[:3]})" if extras else ""))
    print(f"  labelled but missing:     {len(missing)}")

    # ----- Build manifest -----
    print(f"\nBuilding manifest ...")
    manifest_rows = []
    n_processed = 0
    n_skipped_view = 0
    n_skipped_fen = 0
    for row in label_rows:
        image_name = Path(row["image_path"]).name
        fen = row["fen"]
        view = view_from_camera_field(row["camera"])

        if view not in VALID_VIEWS:
            print(f"  skip {image_name}: unknown view {view!r}")
            n_skipped_view += 1
            continue

        try:
            grid = fen_to_label_grid(fen, view)
        except ValueError as e:
            print(f"  skip {image_name}: invalid FEN ({e})")
            n_skipped_fen += 1
            continue

        for r in range(8):
            for c in range(8):
                manifest_rows.append({
                    "source_image": image_name,
                    "view": view,
                    "board_row": r,
                    "board_col": c,
                    "label": int(grid[r, c]),
                    "fen": fen,
                })
        n_processed += 1

    print(f"\nProcessed: {n_processed}   "
          f"skipped (unknown view): {n_skipped_view}   "
          f"skipped (bad FEN): {n_skipped_fen}")

    # ----- Assertions -----
    expected_total = n_processed * 64
    assert len(manifest_rows) == expected_total, (
        f"row count mismatch: got {len(manifest_rows)}, expected {expected_total}"
    )

    # Exactly 64 rows per image
    per_image = Counter(r["source_image"] for r in manifest_rows)
    bad = [(k, v) for k, v in per_image.items() if v != 64]
    assert not bad, f"images with != 64 rows: {bad[:5]}"

    print(f"\nManifest:        {len(manifest_rows):>8d} rows")
    print(f"Brief estimate:  {6135 * 64:>8d} rows  (6135 × 64)")
    print(f"Actual expected: {6132 * 64:>8d} rows  (6132 labelled × 64)")
    print(f"Matches actual:  {'YES' if len(manifest_rows) == 6132 * 64 else 'NO'}")

    # ----- Write CSV -----
    print(f"\nWriting {OUT_CSV} ...")
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_image", "view", "board_row",
                                          "board_col", "label", "fen"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"  wrote {len(manifest_rows)} rows ({OUT_CSV.stat().st_size / 1024 / 1024:.1f} MB)")

    # ----- Class distribution -----
    counts = Counter(r["label"] for r in manifest_rows)
    total = sum(counts.values())
    print(f"\nClass distribution (n={total}):")
    print(f"  {'cls':>3s}  {'name':<22s}  {'count':>9s}   {'pct':>7s}")
    for cls in range(13):
        n = counts.get(cls, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<22s}  {n:>9d}   {pct:>6.2f}%  {bar}")

    # ----- Sanity warnings -----
    warns = []
    empty_pct = 100 * counts.get(12, 0) / total
    if not 40 <= empty_pct <= 70:
        warns.append(f"empty cells {empty_pct:.2f}% outside expected [40%, 70%]")
    if counts.get(5, 0) == 0:
        warns.append("zero white kings (class 5)")
    if counts.get(11, 0) == 0:
        warns.append("zero black kings (class 11)")
    # White / black piece counts should be similar (within ~20%) over a big enough corpus
    white_total = sum(counts.get(c, 0) for c in range(6))
    black_total = sum(counts.get(c, 0) for c in range(6, 12))
    if white_total > 0 and black_total > 0:
        ratio = max(white_total, black_total) / min(white_total, black_total)
        if ratio > 1.2:
            warns.append(f"white/black piece imbalance: white={white_total}, "
                         f"black={black_total} (ratio {ratio:.2f})")
    if warns:
        print(f"\n  WARNINGS:")
        for w in warns:
            print(f"    - {w}")
    else:
        print(f"\n  (no distribution warnings)")

    # ----- Per-view balance -----
    view_counts = Counter(r["view"] for r in manifest_rows)
    print(f"\nPer-view balance:")
    for view in VALID_VIEWS:
        n = view_counts[view]
        n_imgs = n // 64
        print(f"  {view:<10s}  {n:>8d} rows  ({n_imgs:>5d} images)")
    vmax = max(view_counts.values())
    vmin = min(view_counts.values())
    if vmin > 0 and vmax / vmin > 1.05:
        print(f"  WARNING: per-view imbalance > 5% (max={vmax}, min={vmin})")
    else:
        print(f"  (per-view balanced within 5%)")


if __name__ == "__main__":
    main()
