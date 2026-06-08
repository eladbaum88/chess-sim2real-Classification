"""
chess_dataset.py — PyTorch Dataset for per-square classification.

One sample = (one source image, one board square) → (crop_tensor, label).

Pipeline inside __getitem__:
  1. Look up manifest row idx → source_image, view, board_row, board_col, label.
  2. cv2.imread the source image  (uint8 BGR, 512×512).
  3. Read cached corners from corners.json dict  (O(1) lookup).
  4. warp_chessboard_image(bgr, corners)  → 500×500 BGR (board at inner [50..450]).
  5. crop_square(warped, row, col)  → 100×100 BGR (2×2 squares centered on target).
  6. Convert BGR → RGB.
  7. Apply self.transform if provided (sees HWC uint8 RGB).
  8. Tensorize: HWC → CHW, uint8 → float32, divide by 255 → values in [0, 1].
  9. Return (tensor[3, 100, 100], label_int).

Augmentation lives in the transform argument — keeping it out of the Dataset
itself lets us ablate it later by swapping in different callables.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/eladbaum/chess_project")
from scripts.verify_woelflein_crops import warp_chessboard_image, crop_square


DEFAULT_DATASET_DIR = Path(
    "/home/eladbaum/chess_project/data_generation/dataset_v1/images"
)


class ChessSquareDataset(Dataset):
    """One sample per (image × board square). 6,132 images × 64 squares =
    392,448 samples in dataset_v1.
    """

    def __init__(
        self,
        manifest,
        corners_json_path,
        dataset_dir=None,
        transform=None,
    ):
        """
        Args:
            manifest: path to manifest.csv OR a pandas DataFrame already
                loaded/filtered. Columns expected:
                source_image, view, board_row, board_col, label, fen.
            corners_json_path: path to corners.json (dict: image_name →
                [[tl_x,tl_y], [tr_x,tr_y], [br_x,br_y], [bl_x,bl_y]]).
            dataset_dir: directory containing source PNGs. Defaults to
                Project2_3/dataset_v1/images/.
            transform: optional callable applied to the (100, 100, 3) uint8
                RGB crop BEFORE tensorization. None for no augmentation.
        """
        if isinstance(manifest, pd.DataFrame):
            self.manifest = manifest.reset_index(drop=True)
        else:
            self.manifest = pd.read_csv(manifest)
        with open(corners_json_path) as f:
            self.corners = json.load(f)
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_name = row["source_image"]
        board_row = int(row["board_row"])
        board_col = int(row["board_col"])
        label = int(row["label"])

        bgr = cv2.imread(str(self.dataset_dir / image_name))
        corners = np.array(self.corners[image_name], dtype=np.float32)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, board_row, board_col)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)

        tensor = torch.from_numpy(np.ascontiguousarray(crop_rgb)) \
                      .permute(2, 0, 1).float() / 255.0

        return tensor, label
