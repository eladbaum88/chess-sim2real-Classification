# Chessboard State Recognition — Synthetic-to-Real

Project 2 (Introduction to Deep Learning, Ben-Gurion University): predict the full
8×8 board state from a single RGB photo of a chessboard, training primarily on
synthetic renders and studying sim-to-real transfer to real chessboard photos.

The pipeline localises the board with a classical corner detector (chesscog), warps
it to a top-down view, crops each of the 64 squares, and classifies every square with
a 13-class CNN/ViT (12 piece types + empty). The graded entry point is
[`evaluation/predict_board.py`](evaluation/predict_board.py).

## Repository layout

| Folder | Contents |
|--------|----------|
| [`evaluation/`](evaluation/) | **Evaluation deliverable.** `predict_board(image)` + the vendored DINOv2 backbone. Loads the graded checkpoint from `checkpoints/` and runs offline. (A local `evaluate.py` validation harness is kept on disk, gitignored.) |
| [`preprocessing/`](preprocessing/) | Shared library: corner detection / warp / crop (`verify_woelflein_crops.py`), FEN→label grid (`fen_to_grid.py`, `view_orientations.py`), the PyTorch `ChessSquareDataset`, manifest/corner-cache builders. |
| [`syn_data_generation/`](syn_data_generation/) | Blender synthetic-dataset generation for **dataset_v1** (`build_dataset_v1.py`, `chess_position_api_v1_hdri.py`, `render_full_dataset_v1.sbatch`) and dataset audits. Runs inside Blender's Python. |
| [`training/dino/`](training/dino/) | DINOv2 ViT-S/14 training code. Shared parametrized trainer in [`training_scripts/train.py`](training/dino/training_scripts/train.py) (6 runs via `--mode`/`--run_name`), plus a per-version folder with the dedicated script for each of the 3 special runs (`dino_combined_Game6boosted`, `dino_realOnly`, `dino_combined_Game6boosted_ablation_LabelSmoothing`). |
| [`checkpoints/`](checkpoints/) | Per-run catalog with metrics ([`checkpoints/README.md`](checkpoints/README.md)). Each run's `best_real.pt` is committed; other variants are gitignored. `evaluation/predict_board.py` loads `dino_combined_Game6boosted/best_real.pt`. |

The ConvNeXt-Tiny and ResNet-18 comparison experiments (`training/convnext/`, `training/resnet18/`)
and all diagnostic figures/results are kept **local only** (gitignored) — they live on the Drive
and in the report PDF, not the repo.

## Setup (from clone)

```bash
git clone <repo-url> chess_project
cd chess_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To run *only* the evaluation deliverable, `pip install -r evaluation/requirements.txt` is enough.

## Inference (the evaluation entry point)

```python
import numpy as np
from evaluation.predict_board import predict_board   # or run from inside evaluation/

board = predict_board(image)   # image: (H, W, 3) RGB uint8 ndarray
# board: torch.Tensor, shape (8, 8), dtype int64, on CPU, values in [0, 12]
#   board[0,0] = top-left square of the image; board[7,7] = bottom-right.
```

Class encoding: `0–5` white P/R/N/B/Q/K, `6–11` black p/r/n/b/q/k, `12` empty.
`predict_board` is deterministic, never raises (returns an all-empty board on hard
failure), and runs offline — see [`evaluation/README.md`](evaluation/README.md) for details.

## Model provenance

The shipped checkpoint is `checkpoints/dino_combined_Game6boosted/best_real.pt` — DINOv2 ViT-S/14 +
linear head, from the `dino_combined_Game6boosted` run (combined synthetic + real training,
epoch 16, selected on game2 real-validation). On **game7** (held entirely out of
training) it scores **per-square 0.9858 / piece-only 0.9708**.

Reproduce that number:

```bash
cd evaluation
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
`checkpoints/` folder (gitignored — only the graded `dino_combined_Game6boosted` weight is committed).

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
`evaluation/dinov2_vendor/`.
