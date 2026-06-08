"""
predict_board.py — Project 2 evaluation entry point.

    predict_board(image: np.ndarray) -> torch.Tensor

Input :  RGB image, shape (H, W, 3), dtype uint8, values [0, 255].
Output:  (8, 8) tensor on CPU, dtype torch.int64, values in [0, 12].
         output[0, 0] = top-left square of the IMAGE.
         output[7, 7] = bottom-right square of the IMAGE.
         (Purely image-based coordinates — no chess-notation assumption.)

Class encoding (Project 2 — 13 classes, NO out-of-distribution class 13):
    0 White Pawn   1 White Rook   2 White Knight 3 White Bishop
    4 White Queen  5 White King   6 Black Pawn   7 Black Rook
    8 Black Knight 9 Black Bishop 10 Black Queen 11 Black King
    12 Empty

Model: DINOv2 ViT-S/14 backbone (vendored, see ./dinov2_vendor) + Linear(384, 13)
head, checkpoint `dino_combined_Game6boosted/best_real.pt` (combined synthetic+real
training). The board is localised with the chesscog corner detector, warped to a
top-down 500x500 view, and each of the 64 squares is classified from a 100x100
crop resized to 224x224 with ImageNet normalisation — exactly the training-time
preprocessing.

Robustness contract: predict_board NEVER raises. Any failure on an individual
image yields a valid all-empty (all 12) (8, 8) int64 CPU tensor instead — a wrong
board is preferable to crashing the grader's evaluation loop.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Make the package portable: resolve everything relative to THIS file so the
# folder runs unchanged from any location (no absolute project paths anywhere).
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "dinov2_vendor") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "dinov2_vendor"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from woelflein_crops import (  # noqa: E402
    find_corners, warp_chessboard_image, crop_square, ChessboardNotLocatedException,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
NUM_CLASSES = 13
EMBED_DIM = 384
EMPTY_CLASS = 12
INPUT = 224                       # ViT-S/14 native (224/14 -> 16x16 = 256 tokens)
SEED = 42                         # RANSAC determinism (matches training/eval path)
CORNER_OOB_TOLERANCE = 8          # px a detected corner may fall outside the frame
CKPT_PATH = os.path.join(_HERE, "checkpoints", "best_real.pt")

_RESIZE = T.Resize((INPUT, INPUT), antialias=True)


def _select_device():
    """Use CUDA only if it ACTUALLY works. `torch.cuda.is_available()` can be
    True while kernel launches fail (e.g. a torch wheel built for a different
    GPU arch than the host's). Probe with the real ops we use (antialiased
    resize + matmul) and fall back to CPU on any error. Output is on CPU either
    way, so this only affects internal compute, not the returned tensor."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.zeros(1, 3, 16, 16, device="cuda")
        _ = _RESIZE(x)                       # exercises _upsample_bilinear2d_aa on CUDA
        _ = (x.flatten(1) @ x.flatten(1).t())  # exercises a CUDA matmul
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


DEVICE = _select_device()
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class DinoClassifier(nn.Module):
    """DINOv2 ViT-S/14 backbone -> CLS embedding (384) -> Linear(384, 13)."""

    def __init__(self, backbone, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


def _build_backbone():
    """Construct the DINOv2 ViT-S/14 backbone with NO network access, using the
    vendored model code. The kwargs replicate torch.hub's `dinov2_vits14(
    pretrained=False)` exactly, so the resulting state-dict keys/shapes match the
    trained checkpoint (verified: strict load succeeds). Falls back to torch.hub
    only if the vendored import is somehow unavailable (needs internet once)."""
    try:
        from dinov2.models.vision_transformer import vit_small
        return vit_small(
            img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
            block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1,
        )
    except Exception:
        # Fallback: fetch the architecture from torch.hub (requires internet on
        # first run; cached afterwards under ~/.cache/torch/hub). pretrained=False
        # because we overwrite all weights with our checkpoint below.
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)


_MODEL = None


def _get_model():
    """Lazily build + load the model once, then cache it for all later calls."""
    global _MODEL
    if _MODEL is None:
        model = DinoClassifier(_build_backbone())
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(DEVICE).eval()
        _MODEL = model
    return _MODEL


# --------------------------------------------------------------------------
# Corner handling
# --------------------------------------------------------------------------
def _full_frame_corners(bgr):
    """Full-frame corners [TL, TR, BR, BL] derived defensively from the image
    shape. NOTE: this fallback assumes the chessboard fills the frame (as in our
    tightly-cropped data); on loosely-framed photos the warp will include
    background and accuracy may degrade — expected, not a bug."""
    h, w = (bgr.shape[0], bgr.shape[1]) if bgr is not None and bgr.ndim >= 2 else (1, 1)
    w = max(int(w), 2)
    h = max(int(h), 2)
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _get_corners(bgr):
    """Detect the board corners; on any failure or out-of-bounds result, fall
    back to full-frame corners. This branch can never raise."""
    h, w = bgr.shape[:2]
    try:
        np.random.seed(SEED)  # RANSAC reproducibility -> deterministic output
        corners = find_corners(bgr)
        lo = -CORNER_OOB_TOLERANCE
        hi_x, hi_y = w + CORNER_OOB_TOLERANCE, h + CORNER_OOB_TOLERANCE
        ok = bool(np.all((corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                         & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)))
        if not ok or corners.shape != (4, 2):
            raise ChessboardNotLocatedException("corners out of bounds")
        return corners.astype(np.float32)
    except Exception:
        return _full_frame_corners(bgr)


# --------------------------------------------------------------------------
# Required entry point
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_board(image: np.ndarray) -> torch.Tensor:
    """Predict the (8, 8) board state from a single RGB uint8 image.

    Always returns a valid (8, 8) int64 CPU tensor with values in [0, 12];
    on any internal failure it returns an all-empty board (all 12)."""
    try:
        # Reproduce the training preprocessing exactly: the model was trained on
        # crops loaded via cv2.imread (BGR) -> find_corners/warp/crop in BGR ->
        # cv2.cvtColor(BGR2RGB). The grader hands us RGB, so convert RGB->BGR first
        # (find_corners also does an internal BGR2GRAY that assumes BGR ordering).
        img = np.ascontiguousarray(image)
        if img.ndim == 2:  # grayscale -> 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

        corners = _get_corners(bgr)
        warped = warp_chessboard_image(bgr, corners)  # 500x500 BGR top-down board

        # Build all 64 crops in image-row-major order (row 0 = top of image).
        crops = []
        for row in range(8):
            for col in range(8):
                crop_bgr = crop_square(warped, row, col)        # 100x100 BGR
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0
                crops.append(t)
        batch = torch.stack(crops, dim=0).to(DEVICE)            # (64, 3, 100, 100)

        # Model boundary: resize to 224 then ImageNet-normalise (training prep).
        batch = _RESIZE(batch)
        batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD

        logits = _get_model()(batch)                            # (64, 13)
        preds = logits.argmax(dim=1).cpu().numpy().reshape(8, 8)
        # No orientation transform needed: crop_square(warped, row, col) is already
        # in image coordinates, so preds[0,0] = top-left, preds[7,7] = bottom-right.
        grid = preds.astype(np.int64)
    except Exception:
        # Hard failure on this image -> safe fallback: all-empty board.
        grid = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)

    out = torch.from_numpy(np.ascontiguousarray(grid)).to(torch.int64).cpu()
    # Final contract guard: shape (8,8), int64, CPU, values clamped to [0, 12].
    if out.shape != (8, 8):
        out = torch.full((8, 8), EMPTY_CLASS, dtype=torch.int64)
    out = out.clamp_(0, EMPTY_CLASS)
    return out


if __name__ == "__main__":
    # Minimal manual check: run on a path passed as argv[1], else a blank image.
    if len(sys.argv) > 1:
        from PIL import Image
        arr = np.array(Image.open(sys.argv[1]).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
    res = predict_board(arr)
    print("shape", tuple(res.shape), "dtype", res.dtype, "device", res.device.type,
          "min", int(res.min()), "max", int(res.max()))
    print(res)
