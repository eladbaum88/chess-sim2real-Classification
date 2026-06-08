"""FEN -> 8x8 label tensor mapping with per-cam_view transform.

The rendered images in dataset_v2 use different camera positions. To make the
per-square classifier work, the label grid must be transformed so that
label[r][c] corresponds to the piece visible at image cell (r, c).

Per-cam_view transform:

  white     -> identity (raw v2 render, camera on white side -> canonical)
  black     -> rot180   (raw v2 render, camera on black side -> 180-rotated)
  east      -> fliplr   (image is horizontally mirrored vs canonical)
  west      -> fliplr   (image is horizontally mirrored vs canonical)
  overhead  -> fliplr   (image is horizontally mirrored vs canonical)
"""
import numpy as np

# Piece -> integer class. Index 12 = empty square.
PIECE_TO_CLASS = {
    "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5,
    "p": 6, "r": 7, "n": 8, "b": 9, "q": 10, "k": 11,
}
CLASS_TO_PIECE = {v: k for k, v in PIECE_TO_CLASS.items()}
CLASS_TO_PIECE[12] = "."  # empty
EMPTY_CLASS = 12
NUM_CLASSES = 13


def _identity(g):
    return g


def _rot180(g):
    return np.rot90(g, 2)


def _fliplr(g):
    return np.fliplr(g)


# Per-cam_view transform applied to the FEN-derived (white-POV) label grid
# to align with the image grid. See module docstring.
VIEW_TRANSFORMS = {
    "white":    _identity,
    "black":    _rot180,
    "east":     _fliplr,
    "west":     _fliplr,
    "overhead": _fliplr,
}


def fen_to_board_tensor(fen: str, cam_view: str) -> np.ndarray:
    """Return an int64 8x8 array where each cell is a class in [0..12].

    Cell (r, c) corresponds to image grid row r (top to bottom) and column c
    (left to right). cam_view determines the transform applied to the FEN
    piece-placement so the labels align with the rendered image.

    Raises:
        KeyError: if cam_view is not in VIEW_TRANSFORMS.
        ValueError: if the FEN piece-placement field is malformed.
    """
    board = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)
    rows = fen.split(" ")[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(rows)}: {fen!r}")
    for r, rank in enumerate(rows):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                if ch not in PIECE_TO_CLASS:
                    raise ValueError(f"Unknown piece {ch!r} in FEN {fen!r}")
                board[r, c] = PIECE_TO_CLASS[ch]
                c += 1
        if c != 8:
            raise ValueError(f"Rank {r} did not sum to 8 in FEN {fen!r}")

    transform = VIEW_TRANSFORMS[cam_view]
    board = transform(board)
    # np.rot90 / np.fliplr return views; copy so callers can write safely.
    return np.ascontiguousarray(board)


def class_grid_to_str(grid: np.ndarray) -> str:
    """Render an 8x8 class grid as a small ASCII board for debugging."""
    lines = []
    for r in range(8):
        line = " ".join(CLASS_TO_PIECE[int(grid[r, c])] for c in range(8))
        lines.append(line)
    return "\n".join(lines)
