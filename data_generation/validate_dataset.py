"""
validate_dataset.py — automated sanity checks for the synthetic dataset.

Catches:
- Missing / truncated / wrong-sized images
- CSV out of sync with disk (missing entries, orphan files)
- Duplicate renders (same image bytes for different FENs)
- FENs with fewer than 3 camera views
- File-size outliers that suggest partial writes

Exits 0 if all checks pass, 1 otherwise.

Usage:
    python validate_dataset.py
"""

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_DIR = Path(__file__).parent / "dataset"
IMAGES_SUBDIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
EXPECTED_SIZE = (512, 512)
EXPECTED_CAMERAS_PER_FEN = 3  # 1_overhead + 2_west + 3_east (for black view)

# Occupancy check: hard floor below which a FEN/image pair is almost certainly
# misaligned. Random FEN/image pairings average ~33/64 (since both sides have
# ~28 occupied squares out of 64). The std-dev heuristic floors out around
# 44/64 on correct pairings (knights/bishops are slim from above), so 40 sits
# safely between "heuristic noise" and "actually mis-paired".
OCCUPANCY_HARD_FLOOR = 40
N_WORST_TO_PRINT = 5


def check(label, ok, detail=""):
    status = "[PASS]" if ok else "[FAIL]"
    tail = f" - {detail}" if detail else ""
    print(f"  {status} {label}{tail}")
    return ok


def fen_to_occupancy(fen_first_field):
    """Parse FEN piece-placement into an 8x8 boolean occupancy grid.

    Row 0 = FEN rank 8 (top from white's POV), col 0 = file a.
    """
    grid = np.zeros((8, 8), dtype=bool)
    ranks = fen_first_field.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen_first_field}")
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                grid[r, c] = True
                c += 1
        if c != 8:
            raise ValueError(f"Rank {r} did not sum to 8: {rank}")
    return grid


def image_to_occupancy(image_path, center_frac=0.5):
    """Estimate per-square occupancy from a rectified 512x512 overhead image.

    Splits into 8x8 squares, takes the central `center_frac` of each square
    (default 50% = 32x32 inner region, which excludes wood-grain texture and
    captures the piece base), computes per-patch grayscale std-dev, and
    Otsu-thresholds the 64 std values into a binary occupancy mask.
    """
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    H, W = arr.shape
    if H % 8 or W % 8:
        raise ValueError(f"Image dims {arr.shape} not divisible by 8")
    sh, sw = H // 8, W // 8
    ch, cw = int(sh * center_frac), int(sw * center_frac)
    pad_h, pad_w = (sh - ch) // 2, (sw - cw) // 2
    stds = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            patch = arr[r * sh + pad_h : r * sh + pad_h + ch,
                        c * sw + pad_w : c * sw + pad_w + cw]
            stds[r, c] = patch.std()

    # Otsu threshold across the 64 std values.
    flat = np.sort(stds.flatten())
    cum = np.cumsum(flat)
    total = cum[-1]
    n = len(flat)
    best_t, best_var = flat[n // 2], -1.0
    for i in range(1, n):
        w_bg = i / n
        w_fg = 1 - w_bg
        mu_bg = cum[i - 1] / i
        mu_fg = (total - cum[i - 1]) / (n - i)
        var = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        if var > best_var:
            best_var = var
            best_t = (flat[i - 1] + flat[i]) / 2
    return stds > best_t


# All 8 dihedral transformations (D4 group) of an 8x8 grid.
DIHEDRAL = [
    ("id",             lambda g: g),
    ("rot90",          lambda g: np.rot90(g, 1)),
    ("rot180",         lambda g: np.rot90(g, 2)),
    ("rot270",         lambda g: np.rot90(g, 3)),
    ("flip_lr",        lambda g: np.fliplr(g)),
    ("flip_ud",        lambda g: np.flipud(g)),
    ("transpose",      lambda g: g.T),
    ("anti_transpose", lambda g: np.rot90(np.fliplr(g), 1)),
]


def main():
    print("=" * 70)
    print(f"Validating dataset at {DATASET_DIR}")
    print("=" * 70)

    if not LABELS_CSV.exists():
        print(f"[FAIL] labels.csv not found at {LABELS_CSV}")
        sys.exit(1)

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nLabels CSV: {len(rows)} rows")
    fen_indices = sorted({int(r["fen_idx"]) for r in rows})
    print(f"Unique fen_idx values: {len(fen_indices)} "
          f"(range {fen_indices[0]}..{fen_indices[-1]})")

    all_ok = True

    # 1. Every listed image exists on disk.
    missing = [r["image_path"] for r in rows
               if not (DATASET_DIR / r["image_path"]).exists()]
    all_ok &= check(f"All {len(rows)} CSV image_paths exist on disk",
                    not missing, f"{len(missing)} missing: {missing[:3]}")

    # 2. No orphan files on disk that aren't in the CSV.
    csv_paths = {r["image_path"] for r in rows}
    on_disk = {f"images/{p.name}" for p in IMAGES_SUBDIR.glob("*.png")}
    orphans = on_disk - csv_paths
    all_ok &= check("No orphan images on disk (not in CSV)",
                    not orphans, f"{len(orphans)} orphans: {sorted(orphans)[:3]}")

    # 3. Every PNG loads and has the expected dimensions.
    bad_dims = []
    unreadable = []
    for r in rows:
        p = DATASET_DIR / r["image_path"]
        if not p.exists():
            continue
        try:
            with Image.open(p) as im:
                im.load()
                if im.size != EXPECTED_SIZE:
                    bad_dims.append((p.name, im.size))
        except Exception as e:
            unreadable.append((p.name, str(e)))
    all_ok &= check("All images loadable as PNG",
                    not unreadable, f"{unreadable[:3]}")
    all_ok &= check(f"All images are {EXPECTED_SIZE}",
                    not bad_dims, f"{bad_dims[:3]}")

    # 4. Each FEN has exactly 3 camera views.
    cams_per_fen = defaultdict(set)
    for r in rows:
        cams_per_fen[int(r["fen_idx"])].add(r["camera"])
    incomplete = [(idx, sorted(cams)) for idx, cams in cams_per_fen.items()
                  if len(cams) != EXPECTED_CAMERAS_PER_FEN]
    all_ok &= check(f"Every FEN has {EXPECTED_CAMERAS_PER_FEN} cameras",
                    not incomplete, f"{incomplete[:3]}")

    # 5. All images unique (no accidental duplicate renders).
    hashes = defaultdict(list)
    for r in rows:
        p = DATASET_DIR / r["image_path"]
        if not p.exists():
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        hashes[h].append(p.name)
    dupes = [v for v in hashes.values() if len(v) > 1]
    all_ok &= check("All images unique (no duplicate bytes)",
                    not dupes, f"{len(dupes)} duplicate groups: {dupes[:2]}")

    # 6. File-size sanity (flag partial writes / weirdness).
    tiny, huge = [], []
    for r in rows:
        p = DATASET_DIR / r["image_path"]
        if not p.exists():
            continue
        sz = p.stat().st_size
        if sz < 30 * 1024:            # <30 KB is suspicious for a 512x512 PNG
            tiny.append((p.name, sz))
        if sz > 3 * 1024 * 1024:       # >3 MB is way larger than expected
            huge.append((p.name, sz))
    all_ok &= check("No suspiciously small images (<30 KB)",
                    not tiny, f"{tiny[:3]}")
    all_ok &= check("No suspiciously large images (>3 MB)",
                    not huge, f"{huge[:3]}")

    # 7. Camera breakdown (informational — confirms naming convention).
    cam_counts = Counter(r["camera"] for r in rows)
    print(f"\nCamera counts: {dict(cam_counts)}")

    # 8. Source-game breakdown (informational).
    game_counts = Counter(r["source_game"] for r in rows)
    print(f"Source-game counts: {dict(game_counts)}")

    # 9. FEN <-> image occupancy: rendered pieces match FEN under one
    #    consistent dihedral orientation per view_perspective.
    print("\n--- FEN <-> image occupancy check (overhead views only) ---")
    overhead_rows = [r for r in rows if r["camera"] == "1_overhead"]
    per_image = []  # (row, scores_by_orient: dict, error: str|"")
    for r in overhead_rows:
        p = DATASET_DIR / r["image_path"]
        if not p.exists():
            continue
        try:
            gt = fen_to_occupancy(r["fen"].split()[0])
            img = image_to_occupancy(p)
        except Exception as e:
            per_image.append((r, {}, str(e)))
            continue
        scores_by_orient = {name: int((fn(img) == gt).sum()) for name, fn in DIHEDRAL}
        per_image.append((r, scores_by_orient, ""))

    # Pick the expected orientation per view as the one that wins (or ties for
    # win) most often — symmetric positions can tie, so use vote-counting that
    # gives credit to all top-scoring orientations.
    orient_votes = defaultdict(Counter)
    for r, sc, err in per_image:
        if not sc:
            continue
        top = max(sc.values())
        for name, m in sc.items():
            if m == top:
                orient_votes[r["view_perspective"]][name] += 1

    expected = {v: c.most_common(1)[0][0] for v, c in orient_votes.items()}
    for v, c in orient_votes.items():
        print(f"  view={v!r}: top-orient votes = {dict(c)} -> expected {expected[v]!r}")

    scores = np.array([max(sc.values()) for _, sc, err in per_image if sc])
    if len(scores):
        print(f"  occupancy match scores (out of 64): "
              f"min={scores.min()} median={int(np.median(scores))} "
              f"mean={scores.mean():.1f} max={scores.max()}")

    below_floor = [(r["image_path"], r["fen"], max(sc.values()))
                   for r, sc, err in per_image
                   if sc and max(sc.values()) < OCCUPANCY_HARD_FLOOR]
    # Inconsistent = expected orientation does NOT tie for top score. Symmetric
    # positions (e.g. start position) tie across multiple orients and pass.
    inconsistent = []
    for r, sc, err in per_image:
        if not sc:
            continue
        exp = expected.get(r["view_perspective"])
        if exp is None:
            continue
        if sc[exp] < max(sc.values()):
            best_name = max(sc, key=sc.get)
            inconsistent.append((r["image_path"], best_name, sc[exp], max(sc.values())))
    errors = [(r["image_path"], err) for r, _, err in per_image if err]

    all_ok &= check(
        "All overhead images parseable (FEN + image)",
        not errors, f"{len(errors)} errors: {errors[:3]}")
    all_ok &= check(
        f"No overhead image scores below hard floor ({OCCUPANCY_HARD_FLOOR}/64) — "
        "below this is almost certainly a mis-paired FEN, not heuristic noise",
        not below_floor,
        f"{len(below_floor)} below floor; first 3: {below_floor[:3]}")
    all_ok &= check(
        "Expected orientation is at least tied for top score on every overhead",
        not inconsistent,
        f"{len(inconsistent)} where expected lost; first 3: {inconsistent[:3]}")

    # Print the worst-N for the user to manually spot-check.
    ranked = sorted(
        [(r, max(sc.values())) for r, sc, err in per_image if sc],
        key=lambda x: x[1],
    )
    if ranked:
        print(f"\n  {N_WORST_TO_PRINT} lowest-scoring overheads (manual spot-check recommended):")
        for r, m in ranked[:N_WORST_TO_PRINT]:
            print(f"    {m}/64  {r['image_path']}  fen={r['fen']!r}")

    print()
    if all_ok:
        print("=" * 70)
        print("ALL AUTOMATED CHECKS PASSED")
        print("=" * 70)
        print("\nNow do a VISUAL spot-check — open any 2-3 of these in an image viewer")
        print("and confirm the pieces match the FEN:\n")
        for idx in (0, len(fen_indices) // 2, len(fen_indices) - 1):
            f_idx = fen_indices[idx]
            sample = next(r for r in rows if int(r["fen_idx"]) == f_idx)
            path = DATASET_DIR / sample["image_path"]
            print(f"  FEN #{f_idx:03d}  ({sample['source_game']} frame {sample['source_frame']})")
            print(f"    {sample['fen']}")
            print(f"    {path}")
    else:
        print("=" * 70)
        print("SOME CHECKS FAILED - see above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
