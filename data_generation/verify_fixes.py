"""Run the 4 reviewer verification renders."""
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).parent.resolve()
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
BASE = PROJECT_DIR / "chess-set.blend"
SCRIPT = PROJECT_DIR / "chess_position_api_v3.py"
HDRI = PROJECT_DIR / "hdris" / "studio_small_03.exr"
OUT = PROJECT_DIR / "dataset_v2" / "verify"
OUT.mkdir(parents=True, exist_ok=True)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
EMPTY = "8/8/8/8/8/8/8/8"
SPARSE = "8/2k5/2p5/2P5/8/3K4/8/6Q1"

CASES = [
    ("01_start_white_set01",  "set_01_original", START, "white"),
    ("02_start_black_set01",  "set_01_original", START, "black"),
    ("03_start_white_set03",  "set_03_chess",    START, "white"),
    ("04_empty_white_set01",  "set_01_original", EMPTY, "white"),
    ("05_sparse_white_set05", "set_05_staunton", SPARSE, "white"),
]


def run(name, chess_set, fen, view):
    out = OUT / f"{name}.png"
    cmd = [str(BLENDER), str(BASE), "--background", "--python", str(SCRIPT), "--",
           "--chess-set", chess_set, "--fen", fen, "--output", str(out),
           "--hdri", str(HDRI), "--cam-view", view, "--cam-angle-deg", "40",
           "--samples", "64"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    el = time.perf_counter() - t0
    ok = p.returncode == 0 and out.exists()
    print(f"\n=== {name}  ({chess_set}, view={view}) ===")
    print(f"  FEN: {fen}")
    print(f"  [{('OK' if ok else 'FAIL')}] {el:.1f}s -> {out.name}")
    for line in (p.stdout or "").splitlines():
        if any(k in line for k in ("library-load", "apply_fen", "FATAL",
                                    "ERROR", "ValueError", "CAMERA",
                                    "spawned", "Render engine", "white=")):
            print(f"    {line}")
    if not ok:
        print(f"    stderr tail: {(p.stderr or '')[-300:]}")
    return ok


def main():
    print(f"Outputs: {OUT}\n")
    n = sum(1 for c in CASES if run(*c))
    print(f"\n{'=' * 50}\nSummary: {n}/{len(CASES)} OK")


if __name__ == "__main__":
    main()
