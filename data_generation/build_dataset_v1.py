"""
Build dataset_v1: continuation of the first 1500-image dataset.

Same Blender chess set, same camera angle (25°), same lens (26), same resolution
(800 raw → 512 rectified), same three views per FEN (overhead, west, east) from
the black perspective. The ONLY difference is per-FEN HDRI lighting variation:
one HDRI is sampled per FEN and used as the world environment.

Designed to run on the Linux cluster GPU (different Blender path than the
Windows v1 driver). FENs are loaded from the same Fens/*.zip course archives,
in the same dedup order as v1, so dataset_v1 indices line up with v1 indices.

Usage:
    # 10 sample FENs (30 images) on the cluster GPU
    python build_dataset_v1.py --limit 10

    # Resume
    python build_dataset_v1.py --resume
"""

import argparse
import csv
import io
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ======================================================================
# CONFIG
# ======================================================================
BLENDER_EXE = Path("/home/eladbaum/blender/blender")
PROJECT_DIR = Path(__file__).parent.resolve()
BLEND_FILE = PROJECT_DIR / "chess-set.blend"
RENDER_SCRIPT = PROJECT_DIR / "chess_position_api_v1_hdri.py"
RENDERS_DIR = PROJECT_DIR / "renders"
DATASET_DIR = PROJECT_DIR / "dataset_v1"
IMAGES_SUBDIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
LOGS_DIR = DATASET_DIR / "logs"
FENS_DIR = PROJECT_DIR / "Fens"
V1_LABELS_CSV = PROJECT_DIR / "dataset" / "labels.csv"
HDRIS_DIR = PROJECT_DIR / "hdris"

HDRIS = [
    HDRIS_DIR / "studio_small_03.exr",
    HDRIS_DIR / "brown_photostudio_02.exr",
    HDRIS_DIR / "lebombo.exr",
    HDRIS_DIR / "entrance_hall.exr",
]

LABELS_HEADER = [
    "fen_idx", "run_idx", "fen", "source_game", "source_frame",
    "view_perspective", "camera", "image_path",
    "hdri", "hdri_rotation_deg", "hdri_strength",
]


def load_fens_from_zipped_csvs(fens_dir):
    """Same dedup logic as v1's build_dataset.py — keeps fen_idx aligned."""
    entries = []
    seen = set()
    for zip_path in sorted(fens_dir.glob("*.zip")):
        game_name = zip_path.stem.replace("_per_frame", "")
        with zipfile.ZipFile(zip_path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                continue
            with z.open(csvs[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    fen = row["fen"].strip()
                    if not fen or fen in seen:
                        continue
                    seen.add(fen)
                    entries.append({
                        "fen": fen,
                        "source_game": game_name,
                        "source_frame": int(row["from_frame"]),
                    })
    return entries


def load_fens_from_v1_labels(labels_csv):
    """Load FENs from v1's dataset/labels.csv, guaranteeing the same
    fen_idx -> fen mapping (and therefore the same characters/positions)
    as the original 1500-image dataset. Deduplicates by fen_idx (which
    appears 3x in v1 labels — one row per view)."""
    entries = []
    seen_idx = set()
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["fen_idx"])
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            entries.append({
                "fen_idx": idx,
                "fen": row["fen"].strip(),
                "source_game": row["source_game"],
                "source_frame": int(row["source_frame"]),
            })
    entries.sort(key=lambda e: e["fen_idx"])
    return entries


def render_fen(fen, fen_idx, run_idx, hdri_path, hdri_rotation, hdri_strength,
               view, resolution, samples, rectified_size, piece_margin,
               outer_padding=0.0, force_cpu=False):
    if RENDERS_DIR.exists():
        for f in RENDERS_DIR.glob("*_rectified.png"):
            f.unlink()
        for f in RENDERS_DIR.glob("*_raw_tmp.png"):
            f.unlink()

    cmd = [
        str(BLENDER_EXE), str(BLEND_FILE), "--background",
        "--python", str(RENDER_SCRIPT), "--",
        "--fen", fen,
        "--view", view,
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--rectified-size", str(rectified_size),
        "--piece-margin", str(piece_margin),
        "--outer-padding", str(outer_padding),
        "--hdri", str(hdri_path),
        "--hdri-rotation", f"{hdri_rotation:.3f}",
        "--hdri-strength", f"{hdri_strength:.3f}",
    ]
    if force_cpu:
        cmd.append("--cpu")

    log_path = LOGS_DIR / f"fen_{fen_idx:04d}_r{run_idx}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=PROJECT_DIR,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - t0

    if rc != 0:
        print(f"    [FAIL] Blender exited {rc} after {elapsed:.1f}s")
        try:
            tail = log_path.read_text(encoding="utf-8").splitlines()[-30:]
            print("    --- log tail ---")
            for line in tail:
                print(f"    {line}")
        except OSError:
            pass
        return []

    produced = sorted(RENDERS_DIR.glob("*_rectified.png"))
    if not produced:
        print(f"    [FAIL] No rectified images produced (see {log_path})")
        return []

    outputs = []
    for src in produced:
        camera_name = src.stem.replace("_rectified", "")
        dest = IMAGES_SUBDIR / f"fen_{fen_idx:04d}_r{run_idx}_{camera_name}.png"
        shutil.move(str(src), str(dest))
        outputs.append((dest, camera_name))

    print(f"    [OK] {len(outputs)} images in {elapsed:.1f}s")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build dataset_v1 (HDRI-lit continuation)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N FENs (e.g. start past v1's 1533).")
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--rectified-size", type=int, default=512)
    parser.add_argument("--piece-margin", type=float, default=0.1)
    parser.add_argument("--outer-padding", type=float, default=0.0,
                        help="Expand rectification quad outward by this "
                             "fraction; shows a border of scene around the "
                             "board. v1 used 0.0 (tight crop).")
    parser.add_argument("--view", type=str, default="black",
                        choices=["white", "black"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for HDRI choice + rotation/strength.")
    parser.add_argument("--runs-per-fen", type=int, default=1,
                        help="Number of independent renders per FEN. Each "
                             "run gets a different HDRI seed so it produces "
                             "a different lighting variant (camera, materials, "
                             "and piece placement are identical).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-v1-labels", action="store_true",
                        help="Load FENs from dataset/labels.csv (v1's CSV) "
                             "instead of the Fens/*.zip archives. Guarantees "
                             "identical fen_idx -> fen mapping to v1.")
    parser.add_argument("--fen-indices", type=str, default=None,
                        help="Comma-separated list of specific fen_idx values "
                             "to render (only valid with --from-v1-labels). "
                             "Use for targeted re-render or stratified sampling.")
    parser.add_argument("--custom-fen", type=str, default=None,
                        help="Render a single arbitrary FEN. The image is "
                             "saved as fen_diag_<tag>_<camera>.png. Combine "
                             "with --custom-tag to control the name.")
    parser.add_argument("--custom-tag", type=str, default="custom",
                        help="Tag used in custom-FEN output filename.")
    args = parser.parse_args()

    must = [(BLENDER_EXE, "Blender"),
            (BLEND_FILE, "chess-set.blend"),
            (RENDER_SCRIPT, "render script"),
            (HDRIS_DIR, "hdris dir")]
    if args.from_v1_labels:
        must.append((V1_LABELS_CSV, "v1 labels.csv"))
    else:
        must.append((FENS_DIR, "Fens dir"))
    for path, label in must:
        if not Path(path).exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)
    missing_hdris = [h for h in HDRIS if not h.exists()]
    if missing_hdris:
        print(f"ERROR: missing HDRIs: {missing_hdris}", file=sys.stderr)
        sys.exit(1)

    IMAGES_SUBDIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Special case: custom single FEN. Render once, then exit. We do this
    # BEFORE touching labels.csv so a diagnostic render never wipes the
    # existing dataset metadata.
    if args.custom_fen:
        rng = random.Random(args.seed)
        hdri_path = rng.choice(HDRIS)
        hdri_rotation = rng.uniform(0.0, 360.0)
        hdri_strength = rng.uniform(0.7, 1.4)
        tag = args.custom_tag
        print(f"[CUSTOM FEN] tag={tag} fen={args.custom_fen}")
        print(f"  hdri={hdri_path.name} rot={hdri_rotation:.0f}° str={hdri_strength:.2f}")

        # Clear leftover rectified files first
        if RENDERS_DIR.exists():
            for f in RENDERS_DIR.glob("*_rectified.png"):
                f.unlink()
            for f in RENDERS_DIR.glob("*_raw_tmp.png"):
                f.unlink()

        cmd = [
            str(BLENDER_EXE), str(BLEND_FILE), "--background",
            "--python", str(RENDER_SCRIPT), "--",
            "--fen", args.custom_fen,
            "--view", args.view,
            "--resolution", str(args.resolution),
            "--samples", str(args.samples),
            "--rectified-size", str(args.rectified_size),
            "--piece-margin", str(args.piece_margin),
            "--outer-padding", str(args.outer_padding),
            "--hdri", str(hdri_path),
            "--hdri-rotation", f"{hdri_rotation:.3f}",
            "--hdri-strength", f"{hdri_strength:.3f}",
        ]
        if args.cpu:
            cmd.append("--cpu")
        log_path = LOGS_DIR / f"diag_{tag}.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, cwd=PROJECT_DIR,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"  [FAIL] Blender rc={rc} — see {log_path}", file=sys.stderr)
            sys.exit(1)
        produced = sorted(RENDERS_DIR.glob("*_rectified.png"))
        for src in produced:
            cam = src.stem.replace("_rectified", "")
            dest = IMAGES_SUBDIR / f"fen_diag_{tag}_{cam}.png"
            shutil.move(str(src), str(dest))
            print(f"  [OK] {dest.name}")
        return

    # Normal path: prepare CSV header / load resume state. Resume tracks
    # (fen_idx, run_idx) pairs so multi-variant runs can skip done work.
    already_done = set()
    csv_exists = LABELS_CSV.exists()
    if args.resume and csv_exists:
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add((int(row["fen_idx"]), int(row.get("run_idx", 0))))
        print(f"Resuming — {len(already_done)} (fen,run) pairs already in CSV")
    if not args.resume or not csv_exists:
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LABELS_HEADER)

    if args.from_v1_labels:
        entries = load_fens_from_v1_labels(V1_LABELS_CSV)
        source_desc = str(V1_LABELS_CSV)
    else:
        entries = load_fens_from_zipped_csvs(FENS_DIR)
        source_desc = str(FENS_DIR)
    if not entries:
        print(f"ERROR: no FENs loaded from {source_desc}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(entries)} unique FENs from {source_desc}")

    if args.fen_indices:
        if not args.from_v1_labels:
            print("ERROR: --fen-indices requires --from-v1-labels", file=sys.stderr)
            sys.exit(1)
        wanted = {int(x) for x in args.fen_indices.split(",") if x.strip()}
        entries = [e for e in entries if e["fen_idx"] in wanted]
        if len(entries) != len(wanted):
            got = {e["fen_idx"] for e in entries}
            missing = wanted - got
            print(f"WARN: requested fen_idx not in v1 labels: {sorted(missing)}",
                  file=sys.stderr)
        print(f"Filtered to {len(entries)} requested FENs")

    if args.start:
        entries = entries[args.start:]
        print(f"Starting from FEN index {args.start} ({len(entries)} remain)")
    if args.limit:
        entries = entries[:args.limit]

    print(f"\nRendering {len(entries)} FENs (view={args.view}, "
          f"resolution={args.resolution}, samples={args.samples})")
    print(f"HDRIs: {[h.name for h in HDRIS]}")
    print(f"Output: {DATASET_DIR}\n")

    rng = random.Random(args.seed)
    total_ok = 0
    total_images = 0
    batch_start = time.perf_counter()

    for local_idx, entry in enumerate(entries):
        # When loading from v1 labels, preserve the original fen_idx so
        # filenames match dataset/images naming. Otherwise compute from
        # local position + --start (z dedup order).
        if "fen_idx" in entry:
            fen_idx = entry["fen_idx"]
        else:
            fen_idx = (args.start or 0) + local_idx
        fen = entry["fen"]

        for run_idx in range(args.runs_per_fen):
            if (fen_idx, run_idx) in already_done:
                print(f"[FEN {fen_idx:04d} run {run_idx}] skip (done)")
                continue

            # Per (fen_idx, run_idx) RNG so resume reproduces the same HDRI
            # choice for any given pair.
            per_run_rng = random.Random(
                args.seed * 1_000_003 + fen_idx * 1000 + run_idx
            )
            hdri_path = per_run_rng.choice(HDRIS)
            hdri_rotation = per_run_rng.uniform(0.0, 360.0)
            hdri_strength = per_run_rng.uniform(0.7, 1.4)

            print(f"[FEN {fen_idx:04d} run {run_idx}] "
                  f"{entry['source_game']} frame {entry['source_frame']} | "
                  f"hdri={hdri_path.name} rot={hdri_rotation:.0f}° "
                  f"str={hdri_strength:.2f}")

            outputs = render_fen(
                fen, fen_idx, run_idx,
                hdri_path=hdri_path,
                hdri_rotation=hdri_rotation,
                hdri_strength=hdri_strength,
                view=args.view,
                resolution=args.resolution,
                samples=args.samples,
                rectified_size=args.rectified_size,
                piece_margin=args.piece_margin,
                outer_padding=args.outer_padding,
                force_cpu=args.cpu,
            )

            if outputs:
                with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for dest_path, cam_name in outputs:
                        rel = dest_path.relative_to(DATASET_DIR).as_posix()
                        writer.writerow([
                            fen_idx, run_idx, fen,
                            entry["source_game"], entry["source_frame"],
                            args.view, cam_name, rel,
                            hdri_path.name,
                            f"{hdri_rotation:.3f}", f"{hdri_strength:.3f}",
                        ])
                total_ok += 1
                total_images += len(outputs)

    elapsed = time.perf_counter() - batch_start
    print(f"\n{'=' * 60}")
    print(f"Done: {total_ok}/{len(entries)} FENs OK ({total_images} images) "
          f"in {elapsed:.1f}s (avg {elapsed / max(total_ok, 1):.1f}s/FEN)")
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Labels:   {LABELS_CSV}")
    print(f"Logs:     {LOGS_DIR}")


if __name__ == "__main__":
    main()
