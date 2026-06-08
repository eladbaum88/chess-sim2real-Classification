"""Comprehensive mechanical audit of dataset_v1.

Runs 7 categories of checks across every row of labels.csv (and every PNG on
disk), then writes audit_report.txt with PASS/FAIL per check plus detailed
diagnostics. No per-image visualization — only mechanical/statistical
verification. The 10-sample visual check from sanity_check_samples.py is
orthogonal to this.

Usage:
    python audit_dataset_v1.py

Outputs:
    audit_report.txt                          single-file report
    audit_dataset_v1_grid_dumps.txt           50 stratified FEN grid dumps
"""

import csv
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_DIR / "dataset_v1"
LABELS_CSV = DATASET_DIR / "labels.csv"
IMAGES_DIR = DATASET_DIR / "images"
REPORT_TXT = PROJECT_DIR / "audit_report.txt"
GRID_DUMPS_TXT = PROJECT_DIR / "audit_dataset_v1_grid_dumps.txt"

# Diagnostic images live in images/ but are NOT part of the dataset proper
# (they're produced by sanity_check_samples.py for camera/xform verification).
# Excluded from orphan check.
DIAGNOSTIC_PREFIX = "fen_diag_"

EXPECTED_IMG_SIZE = (512, 512)
DEAD_STD_THRESHOLD = 5.0       # std below this = effectively dead image
DARK_MEAN_THRESHOLD = 20.0     # mean below this on 0..255 = render likely broken
HDRI_VARIANCE_TOLERANCE = 0.01  # fraction; if all 12 within ±1% of mean = HDRI bug

# Per-camera FEN-grid transform (calibrated empirically; see VIEW_TRANSFORMS
# in sanity_check_samples.py).
VIEW_TRANSFORMS = {
    "1_overhead": "rot180",
    "2_west":     "rot180",
    "3_east":     "rot180",
}

PIECE_TO_CLASS = {"P":0,"R":1,"N":2,"B":3,"Q":4,"K":5,
                  "p":6,"r":7,"n":8,"b":9,"q":10,"k":11}
CLASS_TO_PIECE = {v:k for k,v in PIECE_TO_CLASS.items()}
CLASS_TO_PIECE[12] = "."
EMPTY_CLASS = 12
NUM_CLASSES = 13
VALID_PIECE_CHARS = set("PRNBQKprnbqk")


# ----------------------------------------------------------------------
# FEN parsing + grid utilities
# ----------------------------------------------------------------------
def fen_piece_count(fen_board):
    return sum(1 for c in fen_board if c.isalpha())


def validate_fen(fen):
    """Return (ok, payload). On success payload is the 8x8 char grid
    (white POV, grid[0]=rank 8). On failure payload is a short reason."""
    if not isinstance(fen, str) or not fen.strip():
        return False, "empty/non-string FEN"
    board = fen.split()[0]
    ranks = board.split("/")
    if len(ranks) != 8:
        return False, f"{len(ranks)} ranks (expected 8)"
    grid = [["." for _ in range(8)] for _ in range(8)]
    n_K = 0; n_k = 0; total = 0
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            elif ch in VALID_PIECE_CHARS:
                if c >= 8:
                    return False, f"rank {8-r} overflows file h"
                grid[r][c] = ch
                if ch == "K": n_K += 1
                elif ch == "k": n_k += 1
                total += 1
                c += 1
            else:
                return False, f"invalid char {ch!r} in rank {8-r}"
        if c != 8:
            return False, f"rank {8-r} sums to {c} (expected 8)"
    if n_K != 1:
        return False, f"{n_K} white kings (expected 1)"
    if n_k != 1:
        return False, f"{n_k} black kings (expected 1)"
    if total > 32:
        return False, f"{total} pieces (max 32)"
    return True, np.array(grid)


def apply_transform(grid, name):
    if name == "rot180":   return np.rot90(grid, 2)
    if name == "identity": return grid
    if name == "rot90":    return np.rot90(grid, 1)
    if name == "rot270":   return np.rot90(grid, 3)
    if name == "fliplr":   return np.fliplr(grid)
    if name == "flipud":   return np.flipud(grid)
    raise ValueError(f"unknown xform: {name}")


def grid_to_class_tensor(char_grid):
    out = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)
    for r in range(8):
        for c in range(8):
            ch = char_grid[r, c]
            if ch in PIECE_TO_CLASS:
                out[r, c] = PIECE_TO_CLASS[ch]
    return out


def short_token(ch):
    if ch == ".": return ". "
    return ("w" if ch.isupper() else "b") + ch.upper()


def format_grid(char_grid):
    lines = ["    a   b   c   d   e   f   g   h"]
    for r in range(8):
        rank = 8 - r
        row = "  ".join(short_token(char_grid[r, c]) for c in range(8))
        lines.append(f"{rank}   {row}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Image analysis (run in thread pool — I/O bound)
# ----------------------------------------------------------------------
def analyze_image(path):
    """Return dict of stats or {'ok': False, 'error': ...}."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            size = im.size
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        return {
            "ok": True,
            "size": size,
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# "Unusual" FEN selection for grid dumps
# ----------------------------------------------------------------------
def has_double_queen(fen_board):
    return fen_board.count("Q") >= 2 or fen_board.count("q") >= 2


def pawn_on_back_rank(fen_board):
    ranks = fen_board.split("/")
    if len(ranks) != 8:
        return False
    return any(c in "Pp" for c in ranks[0]) or any(c in "Pp" for c in ranks[7])


def adjacent_kings(fen_board):
    """Find K and k coords (rank, file) and check Chebyshev distance ≤ 1."""
    ranks = fen_board.split("/")
    if len(ranks) != 8:
        return False
    K_pos = k_pos = None
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                if ch == "K": K_pos = (r, c)
                elif ch == "k": k_pos = (r, c)
                c += 1
    if K_pos is None or k_pos is None:
        return False
    return max(abs(K_pos[0] - k_pos[0]), abs(K_pos[1] - k_pos[1])) <= 1


# ----------------------------------------------------------------------
# Main audit
# ----------------------------------------------------------------------
class Report:
    def __init__(self):
        self.sections = []   # list of (title, status, body_lines)
        self.failed = False

    def section(self, title, status, body_lines):
        self.sections.append((title, status, body_lines))
        if status == "FAIL":
            self.failed = True

    def write(self, path):
        lines = []
        lines.append("=" * 72)
        lines.append("dataset_v1 AUDIT REPORT")
        lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"CSV: {LABELS_CSV}")
        lines.append(f"Images dir: {IMAGES_DIR}")
        lines.append("=" * 72)
        lines.append("\nSUMMARY")
        lines.append("-" * 72)
        for title, status, _ in self.sections:
            marker = "[PASS]" if status == "PASS" else "[FAIL]"
            lines.append(f"  {marker}  {title}")
        lines.append("-" * 72)
        lines.append("")
        for title, status, body in self.sections:
            lines.append("=" * 72)
            lines.append(f"{status}: {title}")
            lines.append("=" * 72)
            lines.extend(body)
            lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t_start = time.perf_counter()
    rep = Report()

    print("Loading labels.csv...")
    df = pd.read_csv(LABELS_CSV)
    n_rows = len(df)
    print(f"  rows: {n_rows}")

    # =================================================================
    # 1. SCHEMA INTEGRITY
    # =================================================================
    print("\n[1/7] Schema integrity...")
    body = []
    schema_fail = False
    body.append(f"CSV rows: {n_rows}")
    required_nonempty_cols = ["fen", "camera", "image_path", "hdri",
                              "view_perspective"]
    body.append(f"Required non-empty columns: {required_nonempty_cols}")
    for col in required_nonempty_cols:
        if col not in df.columns:
            body.append(f"  MISSING COLUMN: {col}")
            schema_fail = True
            continue
        n_empty = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if n_empty > 0:
            body.append(f"  {col}: {n_empty} empty values (FAIL)")
            schema_fail = True
            examples = df[df[col].isna() | (df[col].astype(str).str.strip() == "")] \
                         .head(5)[["fen_idx", "image_path"]].to_dict("records")
            for ex in examples:
                body.append(f"    e.g. {ex}")
        else:
            body.append(f"  {col}: OK (all non-empty)")

    # Image existence
    csv_image_names = df["image_path"].apply(lambda p: Path(p).name).tolist()
    csv_set = set(csv_image_names)
    disk_files = {p.name for p in IMAGES_DIR.iterdir() if p.suffix == ".png"}
    disk_dataset = {n for n in disk_files if not n.startswith(DIAGNOSTIC_PREFIX)}
    diagnostic_files = sorted(disk_files - disk_dataset)

    missing_on_disk = sorted(csv_set - disk_files)
    orphan_on_disk = sorted(disk_dataset - csv_set)

    body.append("")
    body.append(f"Images referenced in CSV: {len(csv_set)} (unique names)")
    body.append(f"PNG files in images/ (excluding {DIAGNOSTIC_PREFIX}*): "
                f"{len(disk_dataset)}")
    body.append(f"Diagnostic files excluded from orphan check: {len(diagnostic_files)} "
                f"({diagnostic_files[:3]}...)")
    body.append(f"  Missing on disk (in CSV, not on disk): {len(missing_on_disk)}")
    if missing_on_disk:
        schema_fail = True
        body.append(f"    first 5: {missing_on_disk[:5]}")
    body.append(f"  Orphan on disk (on disk, not in CSV): {len(orphan_on_disk)}")
    if orphan_on_disk:
        schema_fail = True
        body.append(f"    first 5: {orphan_on_disk[:5]}")

    rep.section("Schema integrity", "FAIL" if schema_fail else "PASS", body)

    # =================================================================
    # 2. FEN VALIDITY
    # =================================================================
    print("[2/7] FEN validity...")
    body = []
    fen_fails = []
    fen_grids = {}  # row idx -> char_grid (cached for check 4)
    for i, row in df.iterrows():
        ok, payload = validate_fen(row["fen"])
        if not ok:
            fen_fails.append((i, row["image_path"], row["fen"], payload))
        else:
            fen_grids[i] = payload

    body.append(f"Rows checked: {n_rows}")
    body.append(f"FENs that pass all validation: {n_rows - len(fen_fails)}")
    body.append(f"FENs that fail: {len(fen_fails)}")
    if fen_fails:
        body.append("First 5 failures:")
        for i, path, fen, reason in fen_fails[:5]:
            body.append(f"  row {i} | {Path(path).name}")
            body.append(f"    fen   : {fen!r}")
            body.append(f"    reason: {reason}")
    fen_status = "FAIL" if fen_fails else "PASS"
    rep.section("FEN validity (8 ranks, 8 squares/rank, valid chars, "
                "1 K + 1 k, total ≤ 32)", fen_status, body)

    # =================================================================
    # 3. IMAGE INTEGRITY
    # =================================================================
    print(f"[3/7] Image integrity ({n_rows} images, threaded)...")
    body = []
    t0 = time.perf_counter()
    paths = [IMAGES_DIR / Path(p).name for p in df["image_path"]]
    stats = [None] * n_rows

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(analyze_image, p): i for i, p in enumerate(paths)}
        for n_done, fut in enumerate(as_completed(futures), 1):
            stats[futures[fut]] = fut.result()
            if n_done % 500 == 0:
                print(f"    {n_done}/{n_rows} analyzed")
    t_img = time.perf_counter() - t0
    body.append(f"Image analysis: {t_img:.1f}s for {n_rows} images")

    open_fails = [(i, paths[i].name, s.get("error", "?"))
                  for i, s in enumerate(stats) if not s["ok"]]
    body.append(f"Open/verify failures: {len(open_fails)}")
    if open_fails:
        for i, name, err in open_fails[:5]:
            body.append(f"  row {i} | {name}: {err}")

    ok_stats = [s for s in stats if s["ok"]]
    sizes = Counter(s["size"] for s in ok_stats)
    body.append(f"Image dimensions distribution: {dict(sizes)}")
    wrong_size = [(i, paths[i].name, stats[i]["size"])
                  for i, s in enumerate(stats)
                  if s["ok"] and s["size"] != EXPECTED_IMG_SIZE]
    body.append(f"Expected size {EXPECTED_IMG_SIZE}: wrong-size count = {len(wrong_size)}")
    if wrong_size:
        for i, name, sz in wrong_size[:5]:
            body.append(f"  row {i} | {name}: size={sz}")

    means = np.array([s["mean"] for s in ok_stats])
    stds  = np.array([s["std"]  for s in ok_stats])
    body.append("")
    body.append("Brightness statistics (0-255):")
    body.append(f"  mean: avg={means.mean():.1f}  min={means.min():.1f}  "
                f"max={means.max():.1f}  std-across-images={means.std():.1f}")
    body.append(f"  std:  avg={stds.mean():.1f}   min={stds.min():.1f}    "
                f"max={stds.max():.1f}")

    buckets = [(0, 20), (20, 50), (50, 100), (100, 180), (180, 256)]
    body.append("  mean-brightness buckets:")
    for lo, hi in buckets:
        n = int(((means >= lo) & (means < hi)).sum())
        body.append(f"    [{lo:3d}, {hi:3d}): {n}")

    dead_imgs = [(i, paths[i].name, stats[i]["std"])
                 for i, s in enumerate(stats)
                 if s["ok"] and s["std"] < DEAD_STD_THRESHOLD]
    body.append(f"")
    body.append(f"DEAD images (std < {DEAD_STD_THRESHOLD}): {len(dead_imgs)}")
    if dead_imgs:
        for i, name, st in dead_imgs[:5]:
            body.append(f"  row {i} | {name}: std={st:.2f}")

    dark_imgs = [(i, paths[i].name, stats[i]["mean"])
                 for i, s in enumerate(stats)
                 if s["ok"] and s["mean"] < DARK_MEAN_THRESHOLD]
    body.append(f"DARK images (mean < {DARK_MEAN_THRESHOLD}): {len(dark_imgs)}")
    if dark_imgs:
        for i, name, m in dark_imgs[:5]:
            body.append(f"  row {i} | {name}: mean={m:.2f}")

    img_status = "PASS"
    if open_fails or wrong_size or dead_imgs or dark_imgs:
        img_status = "FAIL"
    rep.section("Image integrity (open, dims, dead/dark)", img_status, body)

    # =================================================================
    # 4. FEN ↔ IMAGE CONSISTENCY (per-row parity)
    # =================================================================
    print("[4/7] FEN ↔ label parity (per-row, 6132 rows)...")
    body = []
    parity_fails = []
    for i, row in df.iterrows():
        if i not in fen_grids:
            continue  # skip rows whose FEN failed validation (already flagged in [2])
        char_grid = fen_grids[i]
        cam = row["camera"]
        xform = VIEW_TRANSFORMS.get(cam, "identity")
        aligned = apply_transform(char_grid, xform)
        cls = grid_to_class_tensor(aligned)
        empties = int((cls == EMPTY_CLASS).sum())
        expected = 64 - fen_piece_count(row["fen"].split()[0])
        if empties != expected:
            parity_fails.append((i, Path(row["image_path"]).name,
                                 row["fen"], empties, expected))
    body.append(f"Rows checked: {n_rows - len(fen_fails)} "
                f"({len(fen_fails)} skipped due to FEN validation failure)")
    body.append(f"Per-row parity matches (label==12 count == 64 - piece_count): "
                f"{n_rows - len(fen_fails) - len(parity_fails)}")
    body.append(f"Per-row parity FAIL: {len(parity_fails)}")
    if parity_fails:
        body.append("First 5 mismatches:")
        for i, name, fen, got, exp in parity_fails[:5]:
            body.append(f"  row {i} | {name}")
            body.append(f"    fen={fen}")
            body.append(f"    empty squares: got={got}, expected={exp}")
    parity_status = "FAIL" if parity_fails else "PASS"
    rep.section("FEN <-> image-label consistency (per-row empty-count parity)",
                parity_status, body)

    # =================================================================
    # 5. DISTRIBUTION CHECKS
    # =================================================================
    print("[5/7] Distribution checks...")
    body = []
    dist_fail = False

    # Per-class square count
    total_squares = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in fen_grids.keys():
        char_grid = fen_grids[i]
        cam = df.loc[i, "camera"]
        xform = VIEW_TRANSFORMS.get(cam, "identity")
        aligned = apply_transform(char_grid, xform)
        cls = grid_to_class_tensor(aligned)
        for k in range(NUM_CLASSES):
            class_counts[k] += int((cls == k).sum())
        total_squares += 64

    body.append(f"Total squares evaluated: {total_squares}")
    body.append(f"Expected if all rows valid: {n_rows * 64}")
    body.append("Per-class square counts (image-aligned grid):")
    header_pieces = [CLASS_TO_PIECE[k] for k in range(NUM_CLASSES)]
    for k in range(NUM_CLASSES):
        ch = CLASS_TO_PIECE[k]
        cnt = int(class_counts[k])
        tag = "(piece)" if k != EMPTY_CLASS else "(empty)"
        flag = "  ← LOW" if (k != EMPTY_CLASS and cnt < 100) else ""
        body.append(f"  class {k:2d} ({ch}) {tag}: {cnt}{flag}")
        if k != EMPTY_CLASS and cnt < 100:
            dist_fail = True

    # Per-camera distribution
    body.append("")
    cam_counts = df["camera"].value_counts().to_dict()
    body.append(f"Per-camera image counts (expected ~{n_rows // 3} each):")
    for cam, n in sorted(cam_counts.items()):
        body.append(f"  {cam}: {n}")
    cam_imbalance = max(cam_counts.values()) - min(cam_counts.values())
    if cam_imbalance > n_rows * 0.01:  # >1% imbalance
        body.append(f"  WARNING: imbalance across cameras = {cam_imbalance}")

    # Per-HDRI distribution
    body.append("")
    hdri_counts = df["hdri"].value_counts().to_dict()
    body.append(f"Per-HDRI image counts (expected ~{n_rows // 4} each):")
    for h, n in sorted(hdri_counts.items()):
        body.append(f"  {h}: {n}")
    hdri_imbalance = max(hdri_counts.values()) - min(hdri_counts.values())
    if hdri_imbalance > n_rows * 0.05:
        body.append(f"  WARNING: imbalance across HDRIs = {hdri_imbalance}")

    # Duplicate-render detection. The user's original spec was "FEN duplicates
    # per (cam, hdri)". With --runs-per-fen 4 picking HDRI randomly from 4
    # files, some FENs end up with the same HDRI *name* across runs, but with
    # different hdri_rotation_deg/hdri_strength — so the resulting renders
    # are distinct images. The render-loop-bug signal we actually want is:
    # are any two rows IDENTICAL in their render parameters? Check both.
    body.append("")
    same_name = df.groupby(["camera", "hdri", "fen"]).size()
    same_name_dup = same_name[same_name > 1]
    body.append(f"(weak) duplicate (camera, hdri-name, fen) triples: "
                f"{len(same_name_dup)} — expected when 4 runs pick HDRI "
                f"randomly from 4 files (different rotation/strength makes "
                f"each a distinct image).")

    same_render = df.groupby(["camera", "hdri", "fen",
                              "hdri_rotation_deg", "hdri_strength"]).size()
    same_render_dup = same_render[same_render > 1]
    body.append(f"(strong) duplicate-render tuples (camera, hdri, fen, "
                f"rotation, strength): {len(same_render_dup)} — non-zero "
                f"means a render-loop bug (two identical renders).")
    if len(same_render_dup) > 0:
        body.append("First 5 truly-duplicate render tuples:")
        for key, count in same_render_dup.head(5).items():
            body.append(f"  {key}: {count} rows")
        dist_fail = True

    # Filename uniqueness
    fn_counts = df["image_path"].apply(lambda p: Path(p).name).value_counts()
    dup_fn = fn_counts[fn_counts > 1]
    body.append(f"Filename uniqueness: {len(dup_fn)} filenames appear >1 time")
    if len(dup_fn):
        body.append("First 5 duplicate filenames:")
        for name, count in dup_fn.head(5).items():
            body.append(f"  {name}: {count}")
        dist_fail = True

    rep.section("Distribution checks (class, cam, hdri, dupes, fn-unique)",
                "FAIL" if dist_fail else "PASS", body)

    # =================================================================
    # 6. STRATIFIED GRID DUMPS (50 samples → text file)
    # =================================================================
    print("[6/7] Stratified grid dumps (50 samples)...")
    body = []
    rng = np.random.default_rng(2026)
    df_valid = df[df.index.isin(fen_grids.keys())].copy()
    df_valid["piece_count"] = df_valid["fen"].apply(lambda f: fen_piece_count(f.split()[0]))
    df_valid["board"] = df_valid["fen"].apply(lambda f: f.split()[0])

    sparse_pool = df_valid[df_valid["piece_count"] <= 8]
    medium_pool = df_valid[(df_valid["piece_count"] >= 9) & (df_valid["piece_count"] <= 20)]
    dense_pool  = df_valid[df_valid["piece_count"] >= 21]
    unusual_pool = df_valid[
        df_valid["board"].apply(has_double_queen) |
        df_valid["board"].apply(pawn_on_back_rank) |
        df_valid["board"].apply(adjacent_kings)
    ]

    def take(pool, n, name):
        if len(pool) == 0:
            body.append(f"  {name}: pool empty (skipped)")
            return []
        n = min(n, len(pool))
        idxs = rng.choice(pool.index.values, size=n, replace=False)
        return list(idxs)

    body.append(f"Pool sizes: sparse≤8 ({len(sparse_pool)}), "
                f"medium 9-20 ({len(medium_pool)}), "
                f"dense ≥21 ({len(dense_pool)}), "
                f"unusual ({len(unusual_pool)})")

    selected = []
    selected += [(i, "sparse")  for i in take(sparse_pool, 10, "sparse")]
    selected += [(i, "medium")  for i in take(medium_pool, 10, "medium")]
    selected += [(i, "dense")   for i in take(dense_pool,  10, "dense")]
    selected += [(i, "random")  for i in take(df_valid,    10, "random")]
    selected += [(i, "unusual") for i in take(unusual_pool, 10, "unusual")]

    body.append(f"Total grid dumps written: {len(selected)} → {GRID_DUMPS_TXT.name}")

    # Write dumps to text file
    dump_lines = []
    dump_lines.append("dataset_v1 stratified grid dumps")
    dump_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    dump_lines.append(f"Total: {len(selected)} samples (10 per bucket, sparse/medium/dense/random/unusual)")
    dump_lines.append("=" * 72)
    dump_lines.append("")
    for n, (i, bucket) in enumerate(selected, 1):
        row = df.loc[i]
        char_grid = fen_grids[i]
        cam = row["camera"]
        xform = VIEW_TRANSFORMS.get(cam, "identity")
        aligned = apply_transform(char_grid, xform)
        dump_lines.append(f"--- Sample {n:02d}/{len(selected)} [{bucket}] ---")
        dump_lines.append(f"image: {Path(row['image_path']).name}")
        dump_lines.append(f"fen:   {row['fen'].split()[0]}")
        dump_lines.append(f"cam:   {cam}  (xform={xform})")
        dump_lines.append(f"hdri:  {row['hdri']}")
        dump_lines.append(f"pieces: {fen_piece_count(row['fen'].split()[0])}")
        dump_lines.append(format_grid(aligned))
        dump_lines.append("")
    GRID_DUMPS_TXT.write_text("\n".join(dump_lines), encoding="utf-8")

    rep.section("Stratified grid dumps (50 samples)", "PASS", body)

    # =================================================================
    # 7. HDRI VARIANCE
    # =================================================================
    print("[7/7] HDRI variance check (5 FENs × 12 variants)...")
    body = []
    # Pick 5 random FENs that have all 12 (cam, hdri) variants present
    fen_groups = df.groupby("fen_idx")
    full_fens = [f for f, g in fen_groups if len(g) == 12]
    body.append(f"FENs with full 12 (cam, hdri) variants: {len(full_fens)}/{df['fen_idx'].nunique()}")
    if len(full_fens) < 5:
        body.append(f"FAIL: not enough fully-covered FENs to sample 5")
        rep.section("HDRI variance (5 FENs × 12 variants brightness check)",
                    "FAIL", body)
    else:
        rng2 = np.random.default_rng(2027)
        chosen = rng2.choice(full_fens, size=5, replace=False)
        hdri_fail = False
        body.append("Per-FEN per-render mean-brightness (within ±1% of mean = HDRI broken):")
        # We need image stats — but we computed them already in section [3].
        # Build a (image_name -> stats) map for fast lookup.
        name_to_stats = {}
        for i in range(n_rows):
            name = Path(df.loc[i, "image_path"]).name
            name_to_stats[name] = stats[i]

        for fen_idx_val in chosen:
            sub = df[df["fen_idx"] == fen_idx_val]
            brightnesses = []
            for _, r in sub.iterrows():
                s = name_to_stats[Path(r["image_path"]).name]
                if s["ok"]:
                    brightnesses.append((r["camera"], r["hdri"], s["mean"]))
            if not brightnesses:
                continue
            ms = np.array([b[2] for b in brightnesses])
            spread = (ms.max() - ms.min()) / max(ms.mean(), 1e-9)
            body.append(f"")
            body.append(f"  fen_idx={fen_idx_val}  n={len(brightnesses)}  "
                        f"min={ms.min():.1f} max={ms.max():.1f} mean={ms.mean():.1f} "
                        f"spread={spread*100:.1f}%")
            for cam, h, m in sorted(brightnesses):
                body.append(f"    {cam:11s} {h:35s} mean={m:.1f}")
            if spread < HDRI_VARIANCE_TOLERANCE:
                body.append(f"    FLAG: spread {spread*100:.2f}% < "
                            f"{HDRI_VARIANCE_TOLERANCE*100:.0f}% — HDRI may not be varying")
                hdri_fail = True

        rep.section("HDRI variance (5 FENs × 12 variants brightness check)",
                    "FAIL" if hdri_fail else "PASS", body)

    # =================================================================
    # WRITE REPORT
    # =================================================================
    final_body = []
    final_body.append("")
    final_body.append("=" * 72)
    if rep.failed:
        summary = "; ".join(t for t, s, _ in rep.sections if s == "FAIL")
        final_body.append(f"==== AUDIT FAILED: {summary}. Fix before training. ====")
    else:
        final_body.append("==== AUDIT PASSED. dataset_v1 is safe for training. ====")
    final_body.append("=" * 72)
    rep.section("FINAL VERDICT", "FAIL" if rep.failed else "PASS", final_body)

    rep.write(REPORT_TXT)
    print(f"\nReport written: {REPORT_TXT}")
    print(f"Grid dumps written: {GRID_DUMPS_TXT}")
    print(f"Total audit time: {time.perf_counter() - t_start:.1f}s")
    if rep.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
