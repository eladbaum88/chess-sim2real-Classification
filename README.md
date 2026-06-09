# Synthetic-to-Real Chessboard State Recognition


## Overview

Given one RGB image of a chessboard, the model returns an 8×8 grid naming the piece on
each square (or *empty*). 

### Steps of action 
1. **Locate the board** - finding the four board corners.
2. **Warp** - the board is perspective-warped to a clean top-down 500×500 view.
3. **Split** - the view is cut into 64 per-square crops.
4. **Classify** - a DINOv2 ViT-S/14 backbone with a small linear head labels each crop into one of 13 classes.
5. **Assemble** - the 64 predictions are stacked back into the 8×8 board.

We explore **sim-to-real transfer**: the classifier is trained on synthetic Blender renders
and evaluated on real chessboard photos under three main setups:
- **zero-shot** (synthetic only)
- **fine-tuned** (adapted on a little real data)
- **combined** (trained jointly on both).

We also compare backbones (DINOv2 vs. ConvNeXt vs. ResNet-18) and run ablations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Project Goals

The point we would like to explore is **data efficiency through synthetic data**:  
- collecting and hand-labelling thousands of real chessboard photos is slow and expensive, whereas synthetic boards can be rendered in bulk, perfectly labelled, for free.  
We explored how far synthetic data alone can go, and how cheaply a little real data closes the rest of the gap.

- Train a per-square piece classifier **primarily on synthetic** renders.
- Measure the **domain gap**: how well a synth-only model transfers to real photos (zero-shot).
- Show that **fine-tuning** or **combining** with only a handful of real frames recovers near-perfect accuracy - cheaper than building a large real dataset from scratch.
- Compare backbones.
- Deliver a single `predict_board(image)` that turns any board photo into a board state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Repository Structure

The repository is organised by function — one folder per stage of the pipeline:

- **`evaluation/`** - the evaluation deliverable: `predict_board(image)` (the entry point), the board-localisation / warp / crop code, the vendored DINOv2 backbone, and a batch evaluator.
- **`preprocessing/`** - the shared data pipeline: board localisation, FEN → label-grid conversion, and the PyTorch per-square dataset.
- **`syn_data_generation/`** - the Blender scripts that render the synthetic dataset (`dataset_v1`).
- **`training/dino/`** - the DINOv2 training code: one parametrised script for the three transfer setups, plus dedicated scripts for the ablations.
- **`checkpoints/`** - a catalogue of every training run; each run's `best_real.pt` is committed, including the shipped model `dino_combined_Game6boosted/best_real.pt`.
- **`demo/`** - a small script to run the model on your own images.

```text
chess_project/
├── README.md
├── requirements.txt
├── evaluation/
│   ├── __init__.py
│   ├── predict_board.py
│   ├── woelflein_crops.py
│   ├── dinov2_vendor/
│   ├── evaluate.py
│   └── requirements.txt
├── preprocessing/
│   ├── verify_woelflein_crops.py
│   ├── fen_to_grid.py
│   ├── view_orientations.py
│   ├── chess_dataset.py
│   ├── build_manifest.py
│   └── cache_all_corners.py
├── syn_data_generation/
│   ├── build_dataset_v1.py
│   ├── chess_position_api_v1_hdri.py
│   └── render_full_dataset_v1.sbatch
├── training/dino/
│   ├── train.py
│   ├── train_realonly_ablation.py
│   └── train_labelsmooth_ablation.py
├── checkpoints/
│   ├── README.md
│   └── dino_combined_Game6boosted/best_real.pt
└── demo/
    └── demo.py
```

The **ConvNeXt** and **ResNet-18** comparison experiments and all diagnostic figures are kept on the shared Drive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Installation

```bash
git clone <repo-url> chess_project
cd chess_project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That's all you need to run inference, the demo, and `predict_board`. 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Data

The datasets are not stored in the repository because of their size; they are hosted on
Google Drive instead.

> 📦 **Download link** — the synthetic renders and the real game frames can be found here:
> **[Project Dataset (Google Drive)](https://drive.google.com/drive/folders/1OfkS4Q8SwWLqP7k5v-YtivrVwtaaMu9U?usp=drive_link)**

Each dataset follows a simple layout — an `images/` folder and a `gt.csv` with columns
`image_name, fen, view`. You do **not** need any of this to run `predict_board` or the demo
on your own images; the trained weights already ship with the repository.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Inference & Pretrained Model

The trained checkpoint — `checkpoints/dino_combined_Game6boosted/best_real.pt` (a DINOv2
ViT-S/14 backbone with a linear head, trained jointly on synthetic and real data) — **ships
with the repository**, so inference runs **fully offline**, with nothing to download.

```python
import numpy as np
from PIL import Image
from evaluation.predict_board import predict_board   # or run from inside evaluation/

image = np.array(Image.open("your_board.jpg").convert("RGB"), dtype=np.uint8)  # RGB uint8
board = predict_board(image)        # torch.Tensor, shape (8,8), int64, CPU, values 0-12
```

Class ids: `0–5` white P/R/N/B/Q/K, `6–11` black p/r/n/b/q/k, `12` empty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Training

DINOv2 training lives in [`training/dino/`](training/dino/). One parametrized script covers
the three transfer setups; dedicated scripts cover the special runs:

```bash
cd training/dino
python train.py --mode zeroshot --run_name dino_zeroshot     # synthetic only
python train.py --mode fine_tuned   --run_name dino_fine_tuned   # adapt on real
python train.py --mode combined   --run_name dino_combined     # joint synth + real
```

> ⚠️ Training needs the **full local setup** (datasets from the Drive under `data/`, plus the
> shared eval module kept local), not a bare clone. The committed repo ships the **inference**
> path ready to run; training code is included for reference and reproduction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Evaluation Function

The required function lives in [`evaluation/predict_board.py`](evaluation/predict_board.py):

```python
def predict_board(image: np.ndarray) -> torch.Tensor
```

It takes an RGB `uint8` image `(H, W, 3)` and returns a CPU `int64` tensor of shape `(8, 8)`
with image-based coordinates (`output[0,0]` is the top-left square of the image) and values
in `[0, 12]`. It is deterministic and never raises — on a hard failure it returns an
all-empty board. You can run it on **any** chessboard photo, including your own.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Demo

Run the model on your own image (or a folder of images):

```bash
python demo/demo.py --input path/to/your_board.jpg
python demo/demo.py --input path/to/folder --save      # --save also writes a PNG
```

It prints the predicted board as an ASCII diagram and the raw `(8, 8)` tensor, and with
`--save` writes a side-by-side `input vs. predicted board` PNG next to each image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Reproducing Results

The held-out result (real game *game7*, never seen in training) is reproduced with the
batch evaluator against ground-truth FENs:

```bash
cd evaluation
python evaluate.py --gt ../data/game7_per_frame/gt.csv \
                   --imgs ../data/game7_per_frame/images --view game7
# -> per-square ≈ 0.9858, piece-only ≈ 0.9708
```

(Requires the `game7` frames from the Drive under `data/`.) To reproduce on your own images,
just use the demo above with the shipped weights.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Method Reference

Board localisation follows Wölflein & Arandjelović, *Determining Chess Game State From an
Image* (J. Imaging 2021) — [chesscog](https://github.com/georg-wolflein/chesscog) (MIT).
The backbone is Meta's DINOv2 (Apache-2.0); a minimal copy is vendored under
`evaluation/dinov2_vendor/`.
