"""
view_orientations.py — confirmed FEN→image-grid transforms per view for
dataset_v1 (Project2_3/dataset_v1/, rendered with chess_position_api_v3.py).

Determined experimentally on 2026-05-20 via 9 orientation diagnostics
(3 FENs × 3 views — opening, midgame, endgame). The user confirmed the same
transform applies in every panel of every view: np.rot90(raw_board, 2).

Why all three views share one transform: chess_position_api_v3.py uses
order_by_world_xy when rectifying, which orders the source quad by world
position (not image position), so all three camera views land in the same
canonical orientation. Image-top consistently = chess rank 1 (white side),
image-left consistently = chess file h. Mapping FEN-native indexing
(row 0 = rank 8, col 0 = file a) to image indexing flips both axes — rot180.

NOTE: this constants file is dataset_v1-specific. Other datasets in the
project use different rectification:
  - Project2_3/dataset/ (1.5K legacy, v2 api):  rot180 (verified 2026-05-20)
  - data/dataset_v2/    (different rectification): per Project2_3/labels.py
    (fliplr for overhead) — NOT verified for our pipeline; treat with caution.
"""
import numpy as np

# Per-view orientation transforms. Update if a future dataset uses a
# different rectification pipeline.
VIEW_ORIENTATIONS = {
    # dataset_v1 synthetic — confirmed 2026-05-20 (Step 4a)
    "overhead": "rot180",
    "west":     "rot180",
    "east":     "rot180",
    # game7 real images — confirmed (Step 6a) on 3 frames spanning
    # opening / midgame / endgame.  The camera is positioned behind white,
    # so image (0,0) = chess square a8 directly = FEN-native (0,0). No
    # transform needed.
    "game7":    "identity",
    # games 2, 4, 5, 6 real images — confirmed by three independent signals:
    #   (1) model-based test on 5 evenly-spaced midgame frames per game,
    #   (2) wider model-based test on 20 frames per game (all margins +0.12 to +0.24
    #       over the next-best transform; verdict 'identity' on every game),
    #   (3) visual verification on a midgame frame from each game (black back rank
    #       labels match black pieces at top of image; white back rank labels match
    #       white pieces at bottom — consistent with white-behind-camera filming).
    # All four games carry gt.csv view='white' on every frame, matching game7's rule.
    "game2":    "identity",
    "game4":    "identity",
    "game5":    "identity",
    "game6":    "identity",
    # games 8, 9, 10, 11 real images (fine-tuning stage 1+) — view='white' in
    # data/real_labels.csv for every frame, same convention as games 2-7
    # (camera behind white, image (0,0) = chess square a8 = FEN-native (0,0)).
    # No transform needed; smoke test in fine_tuning/stage1_10/train.py
    # visually verifies on the 10 chosen training frames before any training.
    "game8":    "identity",
    "game9":    "identity",
    "game10":   "identity",
    "game11":   "identity",
}

# Convenience standalone constant for game7 (same value as VIEW_ORIENTATIONS["game7"]).
GAME7_ORIENTATION = "identity"


def apply_orientation(raw_board: np.ndarray, view: str) -> np.ndarray:
    """Transform a FEN-native (8, 8) grid into image-aligned coords for `view`.

    Input:  raw_board[r, c] indexed FEN-native (row 0 = rank 8, col 0 = file a).
    Output: grid[r, c] indexed in image coordinates (row 0 = top of image,
            col 0 = left of image).
    """
    transform = VIEW_ORIENTATIONS[view]
    if transform == "identity":
        return raw_board.copy()
    if transform == "fliplr":
        return np.fliplr(raw_board).copy()
    if transform == "flipud":
        return np.flipud(raw_board).copy()
    if transform == "rot180":
        return np.ascontiguousarray(np.rot90(raw_board, 2))
    raise ValueError(f"unknown orientation transform: {transform!r}")
