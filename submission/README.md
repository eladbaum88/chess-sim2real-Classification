# Project 2 — Chessboard State Prediction (submission model)

Self-contained implementation of the required evaluation function:

```python
from predict_board import predict_board   # run from inside this folder
board = predict_board(image)               # image: (H, W, 3) RGB uint8 ndarray
# board: torch.Tensor, shape (8, 8), dtype int64, on CPU, values in [0, 12]
```

## What this does

Given a single RGB photo of a chessboard, `predict_board` returns the 8×8 board
state. It localises the board (chesscog classical corner detector → perspective
warp to a 500×500 top-down view), classifies each of the 64 squares from a
100×100 crop, and assembles the grid in **image coordinates**:

- `board[0, 0]` = top-left square of the image
- `board[7, 7]` = bottom-right square of the image

(Purely image-based; no chess-notation orientation is assumed.)

### Class encoding (Project 2 — 13 classes, **no** OOD/class-13)

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|
| ♙ WP | ♖ WR | ♘ WN | ♗ WB | ♕ WQ | ♔ WK | ♟ BP | ♜ BR | ♞ BN | ♝ BB | ♛ BQ | ♚ BK | empty |

Output values are always in `[0, 12]`; `13`/`14` never appear (those are Project 1 only).

## Model

- **Architecture:** DINOv2 ViT-S/14 backbone (384-d CLS embedding) + `Linear(384, 13)` head (~22M params).
- **Checkpoint:** `checkpoints/best_real.pt` — run `dino_combined_Game6boosted`, epoch 16,
  combined synthetic + real training (dataset_v1 synthetic + real games 4/5/6 + manual
  frames), selected on game2 real-validation accuracy.
- **Preprocessing (exactly as trained):** detect corners → warp → 100×100 per-square
  crop → resize to 224×224 (bilinear, antialias) → ImageNet normalise
  (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`).
- **Held-out accuracy** (game7, never seen in training): per-square **0.9858**,
  piece-only **0.9708**.

### Why RGB→BGR inside the function
The model was trained on crops loaded with `cv2.imread` (BGR) and only converted to
RGB at the very end. The grader passes **RGB**, so `predict_board` converts RGB→BGR
first to reproduce the exact training pixel pipeline (the corner detector also assumes
BGR for its internal grayscale conversion).

## Robustness

- **Never crashes.** Any failure on an individual image returns a valid all-empty
  board (all `12`) rather than raising — a wrong board beats failing the whole run.
- **Deterministic.** `np.random.seed(42)` before corner detection; `model.eval()`
  + `torch.no_grad()`. Same image → same output, on CPU or CUDA.
- **Corner fallback.** If detection fails or returns out-of-bounds corners (>8px), it
  falls back to full-frame corners `[[0,0],[W,0],[W,H],[0,H]]`. This assumes the board
  fills the frame (true for our tightly-cropped data); on loosely-framed photos the
  warp includes background and accuracy may degrade — expected, not a bug.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU is used automatically if available, otherwise CPU. The output tensor is always on CPU.

## Offline / DINOv2 backbone — IMPORTANT

**This folder is fully offline-capable.** The DINOv2 ViT-S/14 architecture is
**vendored** under `dinov2_vendor/dinov2/` (Apache-2.0; see `dinov2_vendor/DINOV2_LICENSE`),
so the model is built with **no network call** and all weights come from the bundled
`checkpoints/best_real.pt`. No `torch.hub` download is needed.

> Fallback only: if the vendored import ever fails, the code falls back to
> `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)`,
> which needs internet **once** to fetch the repo code (cached afterwards under
> `~/.cache/torch/hub`). Under normal operation this path is never taken.

## Files

| File | Purpose |
|------|---------|
| `predict_board.py` | **Required entry point** — `predict_board(image)` + model + preprocessing |
| `woelflein_crops.py` | Corner detection, warp, per-square crop (chesscog port, MIT) |
| `dinov2_vendor/` | Vendored DINOv2 ViT-S/14 model code (offline architecture build) |
| `checkpoints/best_real.pt` | Trained weights (`dino_combined_Game6boosted`) |
| `fen_to_grid.py`, `view_orientations.py` | FEN → label grid (used by `evaluate.py` only) |
| `evaluate.py` | Optional accuracy check against ground-truth FENs |
| `requirements.txt` | Dependencies |

## Reproduce the held-out accuracy

```bash
# from inside this folder, with the repo's data/ available:
python evaluate.py --gt ../data/game7_per_frame/gt.csv \
                   --imgs ../data/game7_per_frame/images --view game7
# expected: per-square ~= 0.9858, piece-only ~= 0.9708
```

## Quick smoke test

```bash
python predict_board.py ../data/game7_per_frame/images/frame_000172.jpg
# prints: shape (8, 8) dtype torch.int64 device cpu min 0 max <=12
```
