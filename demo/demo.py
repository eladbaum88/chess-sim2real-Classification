"""
demo.py — run predict_board on your own chessboard image(s) and print the board.

    python demo/demo.py --input path/to/image.jpg
    python demo/demo.py --input path/to/folder      # every image in the folder
    python demo/demo.py --input image.jpg --save     # also save a PNG visualisation
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
from predict_board import predict_board  # noqa: E402

# class id (0-12) -> display glyph: white = UPPER, black = lower, '.' = empty
GLYPH = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k", "."]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ascii_board(grid):
    return "\n".join(" ".join(GLYPH[int(grid[r, c])] for c in range(8)) for r in range(8))


def collect_images(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if os.path.splitext(f)[1].lower() in IMG_EXT)
    return [path]


def save_png(image, grid, img_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not installed — skipping PNG)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("input")
    ax[0].axis("off")
    ax[1].set_xlim(0, 8)
    ax[1].set_ylim(8, 0)
    ax[1].set_aspect("equal")
    ax[1].set_title("predict_board")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    for r in range(8):
        for c in range(8):
            shade = 0.9 if (r + c) % 2 == 0 else 0.6
            ax[1].add_patch(plt.Rectangle((c, r), 1, 1, color=str(shade)))
            g = GLYPH[int(grid[r, c])]
            if g != ".":
                ax[1].text(c + 0.5, r + 0.5, g, ha="center", va="center", fontsize=14)
    out = os.path.splitext(img_path)[0] + "_predicted.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Run predict_board on an image or folder.")
    ap.add_argument("--input", required=True, help="image file or a folder of images")
    ap.add_argument("--save", action="store_true", help="save a PNG visualisation next to each image")
    args = ap.parse_args()

    for img_path in collect_images(args.input):
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        board = predict_board(image)
        print(f"\n=== {os.path.basename(img_path)} ===")
        print(ascii_board(board.numpy()))
        print("tensor:\n", board)
        if args.save:
            save_png(image, board.numpy(), img_path)


if __name__ == "__main__":
    main()
