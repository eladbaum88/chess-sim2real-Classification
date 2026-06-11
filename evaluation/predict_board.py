"""
predict_board.py — Project 2 evaluation entry point.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Resolve imports/paths relative to this file so the folder is portable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "dinov2_vendor") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "dinov2_vendor"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from woelflein_crops import (  # noqa: E402
    find_corners, warp_chessboard_image, crop_square, ChessboardNotLocatedException,
)

NUM_CLASSES = 13
EMBED_DIM = 384
EMPTY_CLASS = 12
INPUT = 224
SEED = 42                       # RANSAC seed -> deterministic output
CORNER_OOB_TOLERANCE = 8

# Graded weight lives in the repo-level checkpoints/; local copy is a fallback.
_CKPT_CANDIDATES = [
    os.path.join(_HERE, "..", "checkpoints", "dino_combined_Game6boosted", "best_real.pt"),
    os.path.join(_HERE, "checkpoints", "best_real.pt"),
]
CKPT_PATH = next((p for p in _CKPT_CANDIDATES if os.path.exists(p)), _CKPT_CANDIDATES[0])

_RESIZE = T.Resize((INPUT, INPUT), antialias=True)


def _select_device():
    """Use CUDA only if it really works — cuda.is_available() can be True while
    kernels fail (torch built for another GPU arch). Probe, else fall back to CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.zeros(1, 3, 16, 16, device="cuda")
        _ = _RESIZE(x)
        _ = (x.flatten(1) @ x.flatten(1).t())
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


DEVICE = _select_device()
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


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
    """Build DINOv2 ViT-S/14 from the vendored code (no network); kwargs match
    torch.hub's dinov2_vits14(pretrained=False) so the checkpoint loads strict."""
    try:
        from dinov2.models.vision_transformer import vit_small
        return vit_small(
            img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
            block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1,
        )
    except Exception:
        # Fallback: fetch the architecture from torch.hub (needs internet once).
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)


_MODEL = None


def _get_model():
    """Build + load the model once, then cache it."""
    global _MODEL
    if _MODEL is None:
        model = DinoClassifier(_build_backbone())
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(DEVICE).eval()
        _MODEL = model
    return _MODEL


def _full_frame_corners(bgr):
    """Full-frame corners [TL, TR, BR, BL] — fallback assuming the board fills the
    frame (accuracy may drop on loosely-framed photos)."""
    h, w = (bgr.shape[0], bgr.shape[1]) if bgr is not None and bgr.ndim >= 2 else (1, 1)
    w = max(int(w), 2)
    h = max(int(h), 2)
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _get_corners(bgr):
    """Detect board corners; fall back to full-frame on failure/OOB. Never raises."""
    h, w = bgr.shape[:2]
    try:
        np.random.seed(SEED)
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


@torch.no_grad()
def predict_board(image: np.ndarray) -> torch.Tensor:
    """Predict the (8, 8) board state from a single RGB uint8 image.

    Always returns a valid (8, 8) int64 CPU tensor with values in [0, 12];
    on any internal failure it returns an all-empty board (all 12)."""
    try:
        # Match training preprocessing: the model was trained on BGR-loaded crops,
        # so convert the incoming RGB to BGR (find_corners also assumes BGR).
        img = np.ascontiguousarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

        corners = _get_corners(bgr)
        warped = warp_chessboard_image(bgr, corners)

        # 64 crops in image row-major order (row 0 = top).
        crops = []
        for row in range(8):
            for col in range(8):
                crop_bgr = crop_square(warped, row, col)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0
                crops.append(t)
        batch = torch.stack(crops, dim=0).to(DEVICE)

        # Model boundary: resize to 224, then ImageNet-normalise.
        batch = _RESIZE(batch)
        batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD

        logits = _get_model()(batch)
        preds = logits.argmax(dim=1).cpu().numpy().reshape(8, 8)
        # Crops are already image-aligned -> no orientation transform needed.
        grid = preds.astype(np.int64)
    except Exception:
        grid = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)

    out = torch.from_numpy(np.ascontiguousarray(grid)).to(torch.int64).cpu()
    if out.shape != (8, 8):
        out = torch.full((8, 8), EMPTY_CLASS, dtype=torch.int64)
    out = out.clamp_(0, EMPTY_CLASS)
    return out


# Importable helpers for report figures (confusion matrices, calibration).
# Same pipeline as predict_board(), plus per-square softmax. predict_board()
# itself is left untouched (it is the graded entry point).
def build_model(ckpt_path=CKPT_PATH):
    """Build the DINOv2 classifier, load `ckpt_path`, return eval()-mode model."""
    model = DinoClassifier(_build_backbone())
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict_board_proba(image: np.ndarray, model=None):
    """Like predict_board() but also returns per-square softmax.

    Returns (grid (8,8) int64 labels, conf (8,8) float32 max-softmax,
    probs (64,13) float32 row-major). Does NOT swallow exceptions."""
    if model is None:
        model = _get_model()

    img = np.ascontiguousarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

    corners = _get_corners(bgr)
    warped = warp_chessboard_image(bgr, corners)

    crops = []
    for row in range(8):
        for col in range(8):
            crop_bgr = crop_square(warped, row, col)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0
            crops.append(t)
    batch = torch.stack(crops, dim=0).to(DEVICE)
    batch = _RESIZE(batch)
    batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD

    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    grid = preds.cpu().numpy().reshape(8, 8).astype(np.int64)
    conf = conf.cpu().numpy().reshape(8, 8).astype(np.float32)
    probs = probs.cpu().numpy().astype(np.float32)
    return grid, conf, probs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from PIL import Image
        arr = np.array(Image.open(sys.argv[1]).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
    res = predict_board(arr)
    print("shape", tuple(res.shape), "dtype", res.dtype, "device", res.device.type,
          "min", int(res.min()), "max", int(res.max()))
    print(res)
