"""Outer driver for the v3 synthetic chess dataset.

Uses ONLY the v1 chess set (chess-set.blend) as the piece source.
For each render, randomizes HDRI + camera + sun jitter per assets_registry.RANGES.
"""
import argparse
import csv
import io
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))
import assets_registry as registry

BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
BASE_BLEND = PROJECT_DIR / "chess-set.blend"
RENDER_SCRIPT = PROJECT_DIR / "chess_position_api_v3.py"
DATASET_DIR = PROJECT_DIR / "dataset_v2"
IMAGES_DIR = DATASET_DIR / "images"
GT_CSV = DATASET_DIR / "gt.csv"
LOGS_DIR = DATASET_DIR / "logs"
FENS_DIR = PROJECT_DIR / "Fens"

# 'view' column now reflects the actual per-render view (east/west/overhead),
# matching v1's three-view convention.

GT_HEADER = [
    "fen_idx", "run_idx", "fen", "source_game", "source_frame",
    "view", "image_name",
    "random_seed",
    "hdri",
    "cam_view", "cam_height_mul", "cam_angle_deg", "cam_yaw_deg",
    "cam_lens", "cam_roll_deg",
    "sun_energy", "sun_temp_k", "sun_azimuth_jitter", "sun_elevation_jitter",
    "hdri_rotation_deg", "hdri_strength",
]
PROGRESS_LOG = None  # set in main()


def load_fens(dir_):
    entries = []; seen = set()
    for z_path in sorted(dir_.glob("*.zip")):
        game = z_path.stem.replace("_per_frame", "")
        with zipfile.ZipFile(z_path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                continue
            with z.open(csvs[0]) as raw:
                rdr = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in rdr:
                    fen = row["fen"].strip()
                    if not fen or fen in seen:
                        continue
                    seen.add(fen)
                    entries.append({
                        "fen": fen,
                        "source_game": game,
                        "source_frame": int(row["from_frame"]),
                    })
    return entries


def sample_params(rng):
    R = registry.RANGES
    return {
        "hdri":                 rng.choice(registry.HDRIS),
        "hdri_rotation_deg":    rng.uniform(*R["hdri_rotation_deg"]),
        "hdri_strength":        rng.uniform(*R["hdri_strength"]),
        "sun_energy":           rng.uniform(*R["sun_energy"]),
        "sun_temp_k":           rng.uniform(*R["sun_temp_k"]),
        "sun_azimuth_jitter":   rng.uniform(*R["sun_azimuth_jitter"]),
        "sun_elevation_jitter": rng.uniform(*R["sun_elevation_jitter"]),
        # Three views per v1 convention. Distributed roughly evenly.
        "cam_view":             rng.choice(["east", "west", "overhead"]),
        "cam_height_mul":       rng.uniform(*R["cam_height_mul"]),
        "cam_angle_deg":        rng.uniform(*R["cam_angle_deg"]),
        "cam_yaw_deg":          rng.uniform(*R["cam_yaw_deg"]),
        "cam_lens":             rng.uniform(*R["cam_lens"]),
        "cam_roll_deg":         rng.uniform(*R["cam_roll_deg"]),
    }


def render_one(fen, fen_idx, run_idx, params, samples, resolution, rectified_size):
    img_name = f"fen_{fen_idx:05d}_{run_idx}_{params['cam_view']}.png"
    out_path = IMAGES_DIR / img_name
    cmd = [
        str(BLENDER_EXE), str(BASE_BLEND), "--background",
        "--python", str(RENDER_SCRIPT), "--",
        "--fen", fen,
        "--output", str(out_path),
        "--hdri", params["hdri"],
        "--hdri-rotation", f"{params['hdri_rotation_deg']:.3f}",
        "--hdri-strength", f"{params['hdri_strength']:.3f}",
        "--sun-energy", f"{params['sun_energy']:.3f}",
        "--sun-temp-k", f"{params['sun_temp_k']:.1f}",
        "--sun-azimuth-jitter", f"{params['sun_azimuth_jitter']:.3f}",
        "--sun-elevation-jitter", f"{params['sun_elevation_jitter']:.3f}",
        "--cam-height-mul", f"{params['cam_height_mul']:.3f}",
        "--cam-angle-deg", f"{params['cam_angle_deg']:.3f}",
        "--cam-yaw-deg", f"{params['cam_yaw_deg']:.3f}",
        "--cam-lens", f"{params['cam_lens']:.3f}",
        "--cam-roll-deg", f"{params['cam_roll_deg']:.3f}",
        "--cam-view", params["cam_view"],
        "--samples", str(samples),
        "--resolution", str(resolution),
        "--rectified-size", str(rectified_size),
    ]
    log_path = LOGS_DIR / f"fen_{fen_idx:05d}_{run_idx}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=lf, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - t0
    ok = (rc == 0) and out_path.exists()
    return ok, img_name, elapsed


def write_csv_row(writer, fen_idx, run_idx, entry, params, image_name, random_seed):
    writer.writerow([
        fen_idx, run_idx, entry["fen"], entry["source_game"], entry["source_frame"],
        params["cam_view"], image_name,
        random_seed,
        Path(params["hdri"]).name,
        params["cam_view"],
        f"{params['cam_height_mul']:.3f}", f"{params['cam_angle_deg']:.3f}",
        f"{params['cam_yaw_deg']:.3f}",
        f"{params['cam_lens']:.3f}", f"{params['cam_roll_deg']:.3f}",
        f"{params['sun_energy']:.3f}", f"{params['sun_temp_k']:.1f}",
        f"{params['sun_azimuth_jitter']:.3f}", f"{params['sun_elevation_jitter']:.3f}",
        f"{params['hdri_rotation_deg']:.3f}", f"{params['hdri_strength']:.3f}",
    ])


def log_progress(msg):
    """Append a line to dataset_v2/progress.log and echo to stdout."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if PROGRESS_LOG is not None:
        try:
            with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def main():
    global PROGRESS_LOG
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--runs-per-fen", type=int, default=1)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--rectified-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle FEN order before applying --limit, so a small "
                        "test batch spans opening / midgame / endgame instead "
                        "of being all early-game positions.")
    p.add_argument("--resume", action="store_true",
                   help="Append to existing gt.csv and skip (fen_idx, run_idx) "
                        "pairs that already have a row.")
    args = p.parse_args()

    for must, label in [(BLENDER_EXE, "Blender"),
                        (BASE_BLEND, "chess-set.blend"),
                        (RENDER_SCRIPT, "render script")]:
        if not Path(must).exists():
            print(f"ERROR: {label} missing at {must}", file=sys.stderr)
            sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG = DATASET_DIR / "progress.log"

    entries = load_fens(FENS_DIR)
    if not entries:
        print(f"ERROR: no FENs from {FENS_DIR}", file=sys.stderr); sys.exit(1)
    log_progress(f"Loaded {len(entries)} unique FENs")
    if args.shuffle:
        shuffler = random.Random(args.seed)
        shuffler.shuffle(entries)
        log_progress(f"Shuffled FEN order (seed={args.seed})")
    if args.limit:
        entries = entries[:args.limit]
        log_progress(f"Limit: {len(entries)} FENs")

    # Resume mode: keep existing CSV + collect done keys to skip
    done_keys = set()
    csv_exists = GT_CSV.exists()
    if args.resume and csv_exists:
        with open(GT_CSV, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    done_keys.add((int(row["fen_idx"]), int(row["run_idx"])))
                except (KeyError, ValueError):
                    pass
        log_progress(f"Resume: {len(done_keys)} (fen, run) pairs already in gt.csv")
    if not args.resume or not csv_exists:
        with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(GT_HEADER)

    n_target = len(entries) * args.runs_per_fen
    log_progress(f"Target: {n_target} renders ({len(entries)} × {args.runs_per_fen})")
    log_progress(f"HDRIs: {[Path(h).name for h in registry.HDRIS]}")
    log_progress(f"Output: {DATASET_DIR}")

    ok_count = 0; fail_count = 0; skip_count = 0
    t_start = time.perf_counter()
    for fen_idx, entry in enumerate(entries):
        for run_idx in range(args.runs_per_fen):
            per_image = args.seed * 100_000_000 + fen_idx * 100 + run_idx
            if (fen_idx, run_idx) in done_keys:
                skip_count += 1
                continue
            rng = random.Random(per_image)
            params = sample_params(rng)

            # If the output PNG already exists, treat as a resumable skip:
            # write the CSV row (if missing) and continue.
            expected_out = IMAGES_DIR / f"fen_{fen_idx:05d}_{run_idx}_{params['cam_view']}.png"
            if expected_out.exists() and (fen_idx, run_idx) not in done_keys:
                with open(GT_CSV, "a", newline="", encoding="utf-8") as f:
                    write_csv_row(csv.writer(f), fen_idx, run_idx, entry, params,
                                  expected_out.name, per_image)
                skip_count += 1
                continue

            log_progress(f"[fen {fen_idx:05d}/{len(entries)-1} run {run_idx}] "
                         f"{Path(params['hdri']).stem}  view={params['cam_view']}  "
                         f"pitch={params['cam_angle_deg']:.1f}° yaw={params['cam_yaw_deg']:+.1f}°")
            ok, img_name, t = render_one(
                entry["fen"], fen_idx, run_idx, params,
                samples=args.samples, resolution=args.resolution,
                rectified_size=args.rectified_size,
            )
            with open(GT_CSV, "a", newline="", encoding="utf-8") as f:
                write_csv_row(csv.writer(f), fen_idx, run_idx, entry, params,
                              img_name if ok else "", per_image)
            if ok:
                ok_count += 1
                log_progress(f"    [OK] {t:.1f}s -> {img_name}")
            else:
                fail_count += 1
                log_progress(f"    [FAIL] {t:.1f}s  fen={entry['fen']!r}")

    elapsed = time.perf_counter() - t_start
    attempted = ok_count + fail_count
    avg = (elapsed / attempted) if attempted else 0.0
    summary = (f"Done: {ok_count} OK, {fail_count} FAIL, {skip_count} skipped "
               f"in {elapsed:.1f}s (avg {avg:.1f}s/render)")
    log_progress("=" * 60)
    log_progress(summary)


if __name__ == "__main__":
    main()
