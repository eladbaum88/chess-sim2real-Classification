# `predict_board` — Chessboard State Recognition

The inference deliverable. `predict_board` turns a single RGB photo of a chessboard into an
8×8 board state. It loads the trained checkpoint from the repo-level `checkpoints/` folder
(`../checkpoints/dino_combined_Game6boosted/best_real.pt`).

```python
from predict_board import predict_board   # run from inside this folder
board = predict_board(image)               # image: (H, W, 3) RGB uint8 ndarray
# board: torch.Tensor, shape (8, 8), dtype int64, on CPU, values in [0, 12]
```

## What it does

Given one RGB photo, `predict_board` returns the 8×8 board state. It localises the board
(classical corner detector → perspective warp to a 500×500 top-down view), classifies each
of the 64 squares from a 100×100 crop, and assembles the grid in **image coordinates**:

- `board[0, 0]` = top-left square of the image
- `board[7, 7]` = bottom-right square of the image

The mapping is purely image-based — no chess-notation orientation is assumed.

### Class encoding (13 classes)

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|
| ♙ WP | ♖ WR | ♘ WN | ♗ WB | ♕ WQ | ♔ WK | ♟ BP | ♜ BR | ♞ BN | ♝ BB | ♛ BQ | ♚ BK | empty |

Output values are always in `[0, 12]`.

## Model

- **Architecture:** DINOv2 ViT-S/14 backbone (384-d CLS embedding) + `Linear(384, 13)` head (~22M params).
- **Checkpoint:** `../checkpoints/dino_combined_Game6boosted/best_real.pt`, trained jointly on synthetic (`dataset_v1`) and real frames, selected on real-validation accuracy.
- **Preprocessing (identical to training):** detect corners → warp → 100×100 per-square crop
  → resize to 224×224 (bilinear, antialias) → ImageNet normalise
  (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Held-out accuracy** (a real game never seen in training): per-square **0.9858**,
  piece-only **0.9708**.

### Why RGB→BGR inside the function
The model was trained on crops loaded with `cv2.imread` (BGR) and converted to RGB only at
the end. The function receives **RGB**, so it converts RGB→BGR first to reproduce the exact
training pixel pipeline (the corner detector also assumes BGR for its internal grayscale step).

## Robustness

- **Never raises.** Any failure on an individual image returns a valid all-empty board
  (all `12`) instead of throwing - a wrong board is preferable to aborting a whole run.
- **Deterministic.** `np.random.seed(42)` before corner detection, `model.eval()` and
  `torch.no_grad()`. The same image always yields the same output, on CPU or CUDA.
- **Corner fallback.** If detection fails or returns out-of-bounds corners (>8 px), it falls
  back to full-frame corners `[[0,0],[W,0],[W,H],[0,H]]`. This assumes the board fills the
  frame; on loosely-framed photos the warp includes background and accuracy may degrade.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU is used automatically if available, otherwise CPU. The output tensor is always on CPU.

## Offline by design

The DINOv2 ViT-S/14 architecture is **vendored** under `dinov2_vendor/dinov2/` (Apache-2.0;
see `dinov2_vendor/DINOV2_LICENSE`), so the model is built with **no network call** and all
weights come from the committed `../checkpoints/dino_combined_Game6boosted/best_real.pt`.

> Fallback only: if the vendored import ever fails, the code falls back to
> `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)`, which needs
> internet once to fetch the repo code (cached under `~/.cache/torch/hub`). In normal operation
> this path is never taken.

## Files

| File | Purpose |
|------|---------|
| `predict_board.py` | Entry point — `predict_board(image)` + model + preprocessing |
| `woelflein_crops.py` | Corner detection, warp, per-square crop (chesscog port, MIT) |
| `dinov2_vendor/` | Vendored DINOv2 ViT-S/14 model code (offline architecture build) |
| `evaluate.py` | Accuracy against ground-truth FENs (reproduces the held-out number) |
| `fen_to_grid.py`, `view_orientations.py` | FEN → label grid (used by `evaluate.py`) |
| `requirements.txt` | Dependencies |

Weights: `../checkpoints/dino_combined_Game6boosted/best_real.pt` (committed in `checkpoints/`).

## Reproduce the held-out accuracy

```bash
# from inside this folder, with the dataset available under ../data/:
python evaluate.py --gt ../data/game7_per_frame/gt.csv \
                   --imgs ../data/game7_per_frame/images --view game7
# expected: per-square ~= 0.9858, piece-only ~= 0.9708
```

## Quick smoke test

```bash
python predict_board.py ../data/game7_per_frame/images/frame_000172.jpg
# prints: shape (8, 8) dtype torch.int64 device cpu min 0 max <=12
```
