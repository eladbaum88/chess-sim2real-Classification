# Chessboard State Recognition — Synthetic-to-Real

Project 2 (Introduction to Deep Learning, Ben-Gurion University): predict the full
8×8 board state from a single RGB photo of a chessboard, training primarily on
synthetic renders and studying sim-to-real transfer to real chessboard photos.

The pipeline localises the board with a classical corner detector (chesscog), warps
it to a top-down view, crops each of the 64 squares, and classifies every square with
a 13-class CNN/ViT (12 piece types + empty). The graded entry point is
[`submission/predict_board.py`](submission/predict_board.py).

## Repository layout

| Folder | Contents |
|--------|----------|
| [`submission/`](submission/) | **Evaluation deliverable.** Self-contained `predict_board(image)` + the trained checkpoint, vendored DINOv2 backbone, and an `evaluate.py` harness. Runs standalone, offline. |
| [`preprocessing/`](preprocessing/) | Shared library: corner detection / warp / crop (`verify_woelflein_crops.py`), FEN→label grid (`fen_to_grid.py`, `view_orientations.py`), the PyTorch `ChessSquareDataset`, manifest/corner-cache builders. |
| [`data_generation/`](data_generation/) | Blender synthetic-dataset generation (`chess_position_api_v*.py`, `build_dataset_*.py`) and dataset audits. Runs inside Blender's Python. |
| [`training/`](training/) | Model training, one subfolder per architecture: [`dino/`](training/dino/) (DINOv2 ViT-S/14), [`convnext/`](training/convnext/) (ConvNeXt-Tiny), [`resnet18/`](training/resnet18/) (ResNet-18 baselines, fine-tuning and combined stages). Each has its own `README.md`, training scripts, results and plots. |
| [`checkpoints/`](checkpoints/) | Catalog of all trained model runs (one subfolder per run) with metrics. Weight files are hosted externally (Drive) and gitignored; see [`checkpoints/README.md`](checkpoints/README.md). The graded weight is bundled in `submission/`. |
| [`results/`](results/) | Diagnostic figures, metrics, confusion matrices, the written `report/`, and legacy training history. |

## Setup (from clone)

```bash
git clone <repo-url> chess_project
cd chess_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To run *only* the evaluation deliverable, `pip install -r submission/requirements.txt` is enough.

## Inference (the evaluation entry point)

```python
import numpy as np
from submission.predict_board import predict_board   # or run from inside submission/

board = predict_board(image)   # image: (H, W, 3) RGB uint8 ndarray
# board: torch.Tensor, shape (8, 8), dtype int64, on CPU, values in [0, 12]
#   board[0,0] = top-left square of the image; board[7,7] = bottom-right.
```

Class encoding: `0–5` white P/R/N/B/Q/K, `6–11` black p/r/n/b/q/k, `12` empty.
`predict_board` is deterministic, never raises (returns an all-empty board on hard
failure), and runs offline — see [`submission/README.md`](submission/README.md) for details.

## Model provenance

The shipped checkpoint is `submission/checkpoints/best_real.pt` — DINOv2 ViT-S/14 +
linear head, from the `dino_combined_Game6boosted` run (combined synthetic + real training,
epoch 16, selected on game2 real-validation). On **game7** (held entirely out of
training) it scores **per-square 0.9858 / piece-only 0.9708**.

Reproduce that number:

```bash
cd submission
python evaluate.py --gt ../data/game7_per_frame/gt.csv \
                   --imgs ../data/game7_per_frame/images --view game7
```

## Training (per architecture)

Each `training/<arch>/` folder is runnable on its own; see its `README.md`. Example
(DINOv2):

```bash
cd training/dino
python training_scripts/train.py --mode stage5 --run_name dino_combined
python eval_games_2_6.py --run_name dino_combined
```

Training reads the synthetic/real datasets and writes checkpoints under each run's
`checkpoints/` folder (gitignored — only the submission checkpoint is committed).

## Datasets

Datasets are **not** in git (course policy + size). The synthetic renders, the real
per-game frame sets (`game2`–`game11`), and labels live on the shared Google Drive:

> **Datasets:** _<add Google Drive link here>_

Place them under `data/` (real games as `data/game<N>_per_frame/{images,gt.csv}`) to
run training and `evaluate.py`.

## Method reference

Cropping pipeline follows Wölflein & Arandjelović, *Determining Chess Game State From
an Image* (J. Imaging 2021) — [chesscog](https://github.com/georg-wolflein/chesscog)
(MIT). The DINOv2 backbone is Meta's (Apache-2.0); a minimal copy is vendored under
`submission/dinov2_vendor/`.
