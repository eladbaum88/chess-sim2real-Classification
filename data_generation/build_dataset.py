"""
Build a synthetic chess dataset by invoking chess_position_api_v2.py for
a list of FENs.

Architecture: outer driver — this script is a *regular* Python script
(runs in anaconda base, not inside Blender). It spawns `blender.exe` once
per FEN. Each Blender invocation produces 3 rectified images (overhead +
2 angled views). We then move those files into the dataset directory with
a unique name and append rows to labels.csv.

Pros: simple, easy to debug, resumable (CSV is appended incrementally,
so Ctrl+C keeps what's already rendered).
Cons: ~3-5s Blender startup overhead per FEN. Fine for 10-100 FENs; for
1000s, switch to a batched-in-Blender variant.

FENs are loaded from Fens/*.zip (the course-provided game zips). Each zip
has a CSV with columns from_frame, to_frame, fen. We extract every unique
FEN across all games, deduped and ordered by (game, frame).

Usage:
    # Small test (first 3 unique FENs across all game zips)
    python build_dataset.py --limit 3

    # Full run (every unique FEN across all zips)
    python build_dataset.py

    # Resume an interrupted run
    python build_dataset.py --resume
"""

import argparse
import csv
import io
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ======================================================================
# CONFIG
# ======================================================================
BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
PROJECT_DIR = Path(__file__).parent.resolve()
BLEND_FILE = PROJECT_DIR / "chess-set.blend"
RENDER_SCRIPT = PROJECT_DIR / "chess_position_api_v2.py"
RENDERS_DIR = PROJECT_DIR / "renders"         # where Blender writes
DATASET_DIR = PROJECT_DIR / "dataset"         # where we organize the dataset
IMAGES_SUBDIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
LOGS_DIR = DATASET_DIR / "logs"               # per-FEN Blender stdout/stderr
FENS_DIR = PROJECT_DIR / "Fens"               # course-provided game zips


def load_fens_from_zipped_csvs(fens_dir):
    """Load and deduplicate FENs from *.zip files in `fens_dir`.

    Each zip is expected to contain a game CSV with columns
    `from_frame, to_frame, fen` (per the course PDF). Returns a list of
    dicts {fen, source_game, source_frame} in (game, frame) order,
    with each unique FEN kept only at its first occurrence.
    """
    entries = []
    seen_fens = set()

    for zip_path in sorted(fens_dir.glob("*.zip")):
        # "game2_per_frame.zip" → "game2"
        game_name = zip_path.stem.replace("_per_frame", "")
        with zipfile.ZipFile(zip_path) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                print(f"  (warn) no CSV found in {zip_path.name}")
                continue
            with z.open(csv_names[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    fen = row["fen"].strip()
                    if not fen or fen in seen_fens:
                        continue
                    seen_fens.add(fen)
                    entries.append({
                        "fen": fen,
                        "source_game": game_name,
                        "source_frame": int(row["from_frame"]),
                    })
    return entries


# ======================================================================
# RENDERING
# ======================================================================
def render_fen(fen, index, view_perspective, resolution, samples,
               rectified_size, piece_margin, force_cpu=False):
    """Invoke Blender for one FEN. Returns list of (dest_path, camera_name)
    for every image successfully produced, or [] on failure."""
    # Clean any leftover rectified files from previous runs so the globbing
    # below can't pick up stale outputs by accident.
    if RENDERS_DIR.exists():
        for f in RENDERS_DIR.glob("*_rectified.png"):
            f.unlink()
        for f in RENDERS_DIR.glob("*_raw_tmp.png"):
            f.unlink()

    cmd = [
        str(BLENDER_EXE),
        str(BLEND_FILE),
        "--background",
        "--python", str(RENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--view", view_perspective,
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--rectified-size", str(rectified_size),
        "--piece-margin", str(piece_margin),
    ]
    if force_cpu:
        cmd.append("--cpu")

    log_path = LOGS_DIR / f"fen_{index:04d}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_f:
        # Capture combined stdout+stderr to the log file. Don't stream to
        # terminal — it's too noisy for batch runs. On failure we dump the
        # tail to stderr so the user sees what went wrong.
        result = subprocess.run(
            cmd, cwd=PROJECT_DIR,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"    [FAIL] Blender exited {result.returncode} after {elapsed:.1f}s")
        # Show last ~30 lines of log for context
        try:
            tail = log_path.read_text(encoding="utf-8").splitlines()[-30:]
            print("    --- log tail ---")
            for line in tail:
                print(f"    {line}")
            print(f"    --- full log: {log_path} ---")
        except OSError:
            pass
        return []

    # Discover whatever *_rectified.png files Blender produced.
    # View names differ for white/black perspective (e.g. 2_west vs 2_east),
    # so we glob instead of hardcoding names.
    produced = sorted(RENDERS_DIR.glob("*_rectified.png"))
    if not produced:
        print(f"    [FAIL] No rectified images produced (see {log_path})")
        return []

    outputs = []
    for src in produced:
        camera_name = src.stem.replace("_rectified", "")  # e.g. "1_overhead"
        dest = IMAGES_SUBDIR / f"fen_{index:04d}_{camera_name}.png"
        shutil.move(str(src), str(dest))
        outputs.append((dest, camera_name))

    print(f"    [OK] {len(outputs)} images in {elapsed:.1f}s")
    return outputs


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Build synthetic chess dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only render the first N FENs (for small test)")
    parser.add_argument("--resolution", type=int, default=800,
                        help="Blender render resolution (square)")
    parser.add_argument("--samples", type=int, default=128,
                        help="Cycles render quality")
    parser.add_argument("--rectified-size", type=int, default=512,
                        help="Output rectified image size (square)")
    parser.add_argument("--piece-margin", type=float, default=0.1,
                        help="Piece-height lift factor for rectification")
    parser.add_argument("--view", type=str, default="black",
                        choices=["white", "black"],
                        help="Camera perspective")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU rendering (skip GPU probe). Use this "
                             "when CUDA/OPTIX kernel load fails due to old drivers.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip FENs already present in labels.csv")
    args = parser.parse_args()

    # -------- sanity checks --------
    for must_exist, label in [(BLENDER_EXE, "Blender"),
                              (BLEND_FILE, "chess-set.blend"),
                              (RENDER_SCRIPT, "chess_position_api_v2.py")]:
        if not Path(must_exist).exists():
            print(f"ERROR: {label} not found at {must_exist}", file=sys.stderr)
            sys.exit(1)

    # -------- prepare output dirs --------
    IMAGES_SUBDIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # -------- load already-done FENs if resuming --------
    already_done = set()
    csv_exists = LABELS_CSV.exists()
    if args.resume and csv_exists:
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            already_done = {int(row["fen_idx"]) for row in reader}
        print(f"Resuming — {len(already_done)} FENs already in CSV")

    # Write header if starting fresh
    if not args.resume or not csv_exists:
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fen_idx", "fen", "source_game", "source_frame",
                             "view_perspective", "camera", "image_path"])

    # -------- load FENs from the course-provided ZIPs --------
    if not FENS_DIR.exists():
        print(f"ERROR: {FENS_DIR} not found — put the game_*_per_frame.zip "
              f"files there first", file=sys.stderr)
        sys.exit(1)
    entries = load_fens_from_zipped_csvs(FENS_DIR)
    if not entries:
        print(f"ERROR: no FENs loaded from {FENS_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(entries)} unique FENs from {FENS_DIR}")

    if args.limit:
        entries = entries[:args.limit]

    # -------- render loop --------
    print(f"\nRendering {len(entries)} FENs "
          f"(resolution={args.resolution}, samples={args.samples}, "
          f"view={args.view})")
    print(f"Output: {DATASET_DIR}\n")

    total_ok = 0
    total_images = 0
    batch_start = time.perf_counter()

    for idx, entry in enumerate(entries):
        fen = entry["fen"]
        source_game = entry["source_game"]
        source_frame = entry["source_frame"]

        if idx in already_done:
            print(f"[FEN {idx:03d}/{len(entries) - 1}] skip (done)")
            continue

        print(f"[FEN {idx:03d}/{len(entries) - 1}] "
              f"{source_game} frame {source_frame}: {fen}")
        outputs = render_fen(
            fen, idx,
            view_perspective=args.view,
            resolution=args.resolution,
            samples=args.samples,
            rectified_size=args.rectified_size,
            piece_margin=args.piece_margin,
            force_cpu=args.cpu,
        )

        # Append CSV rows immediately so Ctrl+C is safe
        if outputs:
            with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for dest_path, cam_name in outputs:
                    rel = dest_path.relative_to(DATASET_DIR).as_posix()
                    writer.writerow([idx, fen, source_game, source_frame,
                                     args.view, cam_name, rel])
            total_ok += 1
            total_images += len(outputs)

    elapsed = time.perf_counter() - batch_start
    print(f"\n{'=' * 60}")
    print(f"Done: {total_ok}/{len(entries)} FENs OK "
          f"({total_images} images) in {elapsed:.1f}s "
          f"(avg {elapsed / max(total_ok, 1):.1f}s/FEN)")
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Labels:   {LABELS_CSV}")
    print(f"Logs:     {LOGS_DIR}")


if __name__ == "__main__":
    main()
