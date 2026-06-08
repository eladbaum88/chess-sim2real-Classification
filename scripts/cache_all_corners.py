"""
cache_all_corners.py — run chesscog find_corners over every image in
dataset_v1/, fall back to per-view averaged corners on failure, and save
results to corners.json + corner_detection_log.csv.

Per the Step 3 brief:
  - Pass 1: run find_corners on each image; record success / failure /
    failure_reason. Sanity-check successful detections (in-bounds, area,
    aspect) — failures here count as "bad_geometry".
  - Compute per-view fallback corners as the MEAN of all successful
    detections for that view. These are empirically grounded — not guessed.
  - Pass 2 (in-memory only): for any failure, fill in the per-view mean as
    the corners. No re-running detection.
  - Save corners.json (all images, fallback-filled), corner_detection_log.csv
    (per-image audit trail), and fallback_corners.json (the per-view means
    so they can be reused by other scripts).

Runtime expected ~30-40 min (mean 0.16s/image from robustness check × 11196).
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/eladbaum/chess_project")
from scripts.verify_woelflein_crops import find_corners, ChessboardNotLocatedException


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATASET_DIR = Path("/home/eladbaum/chess_project/data_generation/dataset_v1/images")
OUT_JSON = Path("/home/eladbaum/chess_project/corners.json")
OUT_FALLBACK = Path("/home/eladbaum/chess_project/fallback_corners.json")
OUT_CSV = Path("/home/eladbaum/chess_project/corner_detection_log.csv")
PROGRESS_EVERY = 500
SEED = 0

# Sanity-check thresholds — must match check_detector_robustness.py
CORNER_OOB_TOL = 10
MIN_QUAD_AREA_FRACTION = 0.30
ASPECT_RATIO_RANGE = (0.6, 1.66)

VIEWS = ("overhead", "west", "east")


# --------------------------------------------------------------------------
def quad_sanity(corners, img_shape):
    H, W = img_shape[:2]
    if not np.all((corners[:, 0] >= -CORNER_OOB_TOL)
                  & (corners[:, 0] <= W + CORNER_OOB_TOL)
                  & (corners[:, 1] >= -CORNER_OOB_TOL)
                  & (corners[:, 1] <= H + CORNER_OOB_TOL)):
        return False, "bad_geometry: corner OOB"
    area = cv2.contourArea(corners.astype(np.float32))
    if area < MIN_QUAD_AREA_FRACTION * H * W:
        return False, f"bad_geometry: area frac {area/(H*W):.2f}"
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    ymin, ymax = corners[:, 1].min(), corners[:, 1].max()
    if ymax - ymin <= 0:
        return False, "bad_geometry: zero height"
    ar = (xmax - xmin) / (ymax - ymin)
    if not ASPECT_RATIO_RANGE[0] <= ar <= ASPECT_RATIO_RANGE[1]:
        return False, f"bad_geometry: aspect {ar:.2f}"
    return True, ""


def view_of(filename):
    """fen_XXXX_rY_Z_<view>.png  → <view>"""
    return filename.split("_")[-1].rsplit(".", 1)[0]


def categorize_exception(e):
    """Map an exception to one of the spec's failure_reason categories."""
    if isinstance(e, ChessboardNotLocatedException):
        return "ransac_timeout" if "RANSAC" in str(e) else f"chessboard_not_located"
    return f"exception:{type(e).__name__}"


# --------------------------------------------------------------------------
def detect_pass(images):
    """Pass 1: run find_corners on each image, return dict
    filename → {view, status, corners (or None), runtime_s, failure_reason}."""
    out = {}
    t0 = time.perf_counter()
    n_ok = 0
    for i, img_path in enumerate(images, 1):
        view = view_of(img_path.name)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            out[img_path.name] = dict(
                view=view, status="failed", corners=None,
                runtime_s=0.0, failure_reason="exception:imread_returned_None")
            continue

        np.random.seed(SEED)
        t_img = time.perf_counter()
        try:
            corners = find_corners(bgr)
            dt = time.perf_counter() - t_img
            ok, reason = quad_sanity(corners, bgr.shape)
            if ok:
                out[img_path.name] = dict(
                    view=view, status="detected", corners=corners,
                    runtime_s=dt, failure_reason="")
                n_ok += 1
            else:
                out[img_path.name] = dict(
                    view=view, status="failed", corners=None,
                    runtime_s=dt, failure_reason=reason)
        except Exception as e:
            dt = time.perf_counter() - t_img
            out[img_path.name] = dict(
                view=view, status="failed", corners=None,
                runtime_s=dt, failure_reason=categorize_exception(e))

        if i % PROGRESS_EVERY == 0 or i == len(images):
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta = rate * (len(images) - i)
            print(f"  [{i:5d}/{len(images)}]  elapsed {elapsed/60:5.1f}m  "
                  f"rate {rate*1000:4.0f}ms/img  ETA {eta/60:5.1f}m  "
                  f"detected={n_ok}  failed={i-n_ok}", flush=True)
    return out


def compute_per_view_fallback(detections):
    """Mean corners across successful detections per view."""
    fallback = {}
    for view in VIEWS:
        ok = [d["corners"] for d in detections.values()
              if d["status"] == "detected" and d["view"] == view]
        if not ok:
            raise RuntimeError(f"No successful detections for view {view!r}; "
                               "cannot build empirical fallback.")
        avg = np.mean(np.stack(ok), axis=0)
        std = np.std(np.stack(ok), axis=0)
        fallback[view] = avg
        print(f"  fallback {view}: n={len(ok)}  "
              f"mean=[{', '.join(f'({x:.1f},{y:.1f})' for x, y in avg)}]")
        print(f"           σ=[{', '.join(f'({x:.1f},{y:.1f})' for x, y in std)}]")
    return fallback


def main():
    images = sorted(DATASET_DIR.glob("*.png"))
    print(f"Found {len(images)} images in {DATASET_DIR}")
    print(f"Progress every {PROGRESS_EVERY} images.\n")
    t_overall = time.perf_counter()

    # ---- Pass 1 ----
    print("=== Pass 1: detect ===")
    detections = detect_pass(images)

    # ---- Compute fallback ----
    print("\n=== Computing per-view empirical fallback (mean of successes) ===")
    fallback = compute_per_view_fallback(detections)

    # ---- Pass 2: fill failures ----
    n_fallback = 0
    for d in detections.values():
        if d["corners"] is None:
            d["corners"] = fallback[d["view"]]
            d["status"] = "fallback"
            n_fallback += 1

    # ---- Write fallback corners ----
    with OUT_FALLBACK.open("w") as f:
        json.dump({v: fallback[v].tolist() for v in VIEWS}, f, indent=2)
    print(f"\nWrote {OUT_FALLBACK}")

    # ---- Write corners.json ----
    corners_dict = {name: d["corners"].tolist() for name, d in detections.items()}
    with OUT_JSON.open("w") as f:
        json.dump(corners_dict, f)
    print(f"Wrote {OUT_JSON}  ({len(corners_dict)} entries)")

    # ---- Write CSV log ----
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "view", "source", "failure_reason", "runtime_s",
            "tl_x", "tl_y", "tr_x", "tr_y",
            "br_x", "br_y", "bl_x", "bl_y",
        ])
        for name, d in detections.items():
            c = np.asarray(d["corners"])
            w.writerow([
                name, d["view"], d["status"], d["failure_reason"],
                f"{d['runtime_s']:.3f}",
                f"{c[0,0]:.2f}", f"{c[0,1]:.2f}",
                f"{c[1,0]:.2f}", f"{c[1,1]:.2f}",
                f"{c[2,0]:.2f}", f"{c[2,1]:.2f}",
                f"{c[3,0]:.2f}", f"{c[3,1]:.2f}",
            ])
    print(f"Wrote {OUT_CSV}")

    # ---- Summary ----
    by_view = {v: {"detected": 0, "fallback": 0, "reasons": {}} for v in VIEWS}
    for d in detections.values():
        v = d["view"]
        if d["status"] == "detected":
            by_view[v]["detected"] += 1
        else:
            by_view[v]["fallback"] += 1
            by_view[v]["reasons"][d["failure_reason"]] = \
                by_view[v]["reasons"].get(d["failure_reason"], 0) + 1

    n_total = len(detections)
    total_runtime = time.perf_counter() - t_overall

    print("\n=== Summary ===")
    print(f"Total images: {n_total}")
    print(f"Total runtime: {total_runtime/60:.1f} min  "
          f"({total_runtime/n_total*1000:.0f}ms/image avg)")
    print(f"Detected: {n_total - n_fallback}/{n_total} "
          f"({100*(1 - n_fallback/n_total):.2f}%)")
    print(f"Fallback: {n_fallback}/{n_total} "
          f"({100*n_fallback/n_total:.2f}%)")
    for v in VIEWS:
        b = by_view[v]
        n = b["detected"] + b["fallback"]
        print(f"  {v:8s}: {b['detected']}/{n} detected, {b['fallback']} fallback")
        for reason, count in sorted(b["reasons"].items(), key=lambda x: -x[1]):
            print(f"     {reason}: {count}")


if __name__ == "__main__":
    main()
