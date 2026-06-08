"""
fen_to_grid.py — convert a FEN piece-placement string into an (8, 8) int64
label grid in IMAGE coordinates, applying the per-view orientation transform
from view_orientations.py.

Class encoding (project spec, 13 classes total, no 'occluded' for synthetic):
    0  P  white pawn       6  p  black pawn
    1  R  white rook       7  r  black rook
    2  N  white knight     8  n  black knight
    3  B  white bishop     9  b  black bishop
    4  Q  white queen     10  q  black queen
    5  K  white king      11  k  black king
   12  .  empty
"""
import numpy as np

from view_orientations import apply_orientation


PIECE_TO_CLASS = {
    "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5,
    "p": 6, "r": 7, "n": 8, "b": 9, "q": 10, "k": 11,
}
EMPTY_CLASS = 12


def _parse_fen_to_raw_grid(fen: str) -> np.ndarray:
    """Parse the piece-placement field of a FEN into an 8×8 grid in
    FEN-NATIVE coordinates: row 0 = rank 8 (top of FEN string), col 0 = file
    a. Empty squares are EMPTY_CLASS (12). This is the *un-oriented* grid;
    callers should apply view-specific transforms before treating it as
    image-space.

    Raises ValueError on malformed FEN (wrong number of ranks, unknown
    piece character, or rank that doesn't sum to 8).
    """
    board = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen!r}")
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                if ch not in PIECE_TO_CLASS:
                    raise ValueError(f"Unknown piece {ch!r} in FEN {fen!r}")
                if c >= 8:
                    raise ValueError(
                        f"Rank {r} overflowed 8 columns at piece {ch!r}: {fen!r}")
                board[r, c] = PIECE_TO_CLASS[ch]
                c += 1
        if c != 8:
            raise ValueError(
                f"Rank {r} did not sum to 8 (got {c}): {fen!r}")
    return board


def fen_to_label_grid(fen: str, view: str) -> np.ndarray:
    """
    Convert a FEN string to an 8×8 grid of integer labels (0-12) in
    image-coordinate space for the given view.

    Returns array of shape (8, 8), dtype=int64, where grid[row, col] is the
    label for the square at image position (row, col).

    Label encoding (project spec):
      0  P (white pawn)    1  R (white rook)    2  N (white knight)
      3  B (white bishop)  4  Q (white queen)   5  K (white king)
      6  p (black pawn)    7  r (black rook)    8  n (black knight)
      9  b (black bishop) 10  q (black queen)  11  k (black king)
     12  .  (empty)
    """
    raw = _parse_fen_to_raw_grid(fen)
    return apply_orientation(raw, view).astype(np.int64)
