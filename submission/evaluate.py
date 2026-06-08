"""
evaluate.py — optional sanity check for predict_board.

Runs predict_board on a held-out real game's frames and reports per-square /
piece-only / empty accuracy against the ground-truth FENs. This is the end-to-end
round trip that validates the wiring (notably the RGB->BGR conversion and the
resize/normalise order): if those are wrong, accuracy collapses.

Expected on game7 (truly held out of dino_combindedGame6 training):
    per-square ~= 0.9858,  piece-only ~= 0.9708
matching dino/results/dino_combindedGame6/heldout_game7_eval.json.

Usage:
    python evaluate.py --gt   /path/to/gt.csv \
                       --imgs /path/to/images \
                       --view game7
All arguments are optional; defaults point at the project's game7 frames when run
from inside the repo. Images are loaded as RGB uint8 (the grader's input contract).
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

from predict_board import predict_board, EMPTY_CLASS
from fen_to_grid import fen_to_label_grid

_HERE = os.path.dirname(os.path.abspath(__file__))
# Default to the repo's game7 frames if present (only used for in-repo runs;
# pass --gt/--imgs explicitly when running from a copied/clean location).
_DEFAULT_GT = os.path.join(_HERE, "..", "data", "game7_per_frame", "gt.csv")
_DEFAULT_IMGS = os.path.join(_HERE, "..", "data", "game7_per_frame", "images")

CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK",
               "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=_DEFAULT_GT, help="path to gt.csv (image_name,fen,view)")
    ap.add_argument("--imgs", default=_DEFAULT_IMGS, help="directory of frame images")
    ap.add_argument("--view", default="game7",
                    help="orientation key for fen_to_label_grid (real games: gameN -> identity)")
    args = ap.parse_args()

    with open(args.gt) as f:
        rows = list(csv.DictReader(f))
    print(f"evaluating {len(rows)} frames from {args.gt}")

    all_preds, all_labels = [], []
    for i, r in enumerate(rows):
        img_path = os.path.join(args.imgs, r["image_name"])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # RGB uint8
        pred = predict_board(image).numpy()                                     # (8,8) int64
        label = fen_to_label_grid(r["fen"], args.view)                          # (8,8) int64
        all_preds.append(pred.reshape(-1))
        all_labels.append(label.reshape(-1))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    per_square = float((preds == labels).mean())
    pm = labels != EMPTY_CLASS
    piece_only = float((preds[pm] == labels[pm]).mean()) if pm.any() else float("nan")
    em = labels == EMPTY_CLASS
    empty = float((preds[em] == labels[em]).mean()) if em.any() else float("nan")

    print(f"\n=== {args.view} ({len(rows)} frames, {len(preds)} squares) ===")
    print(f"  per-square : {per_square:.4f}")
    print(f"  piece-only : {piece_only:.4f}")
    print(f"  empty      : {empty:.4f}")
    print("  per-class accuracy:")
    for c in range(13):
        m = labels == c
        acc = float((preds[m] == c).mean()) if m.any() else None
        print(f"    {CLASS_SHORT[c]:>5}: " + ("  n/a" if acc is None else f"{acc:.4f}"))


if __name__ == "__main__":
    main()
