"""STEP-0 CONFIRMATION for the DINOv2 ViT-S/14 backbone (no training, no writes).

Confirms before we build the training runs:
  1. DINOv2 ViT-S/14 loads (torch.hub 'dinov2_vits14', timm fallback). Reports which path.
  2. Head = Linear(384, 13); forward on a 100x100->resize batch -> finite (B, 13).
  3. Token count at --input_size: n patch tokens at 224 (expect 256) and 98 (expect 49).
  4. Param count (expect ~21M; between ResNet 11.2M and ConvNeXt 27.8M).
  5. BatchNorm count (expect 0) + LayerNorm count.
  6. Write-guard fires before any write on a frozen-token run_name.
  7. Verbatim eval-harness imports resolve.
  8. DINO sanity linear-probe: frozen CLS features on a few hundred real squares, fit the
     head (sklearn LogisticRegression), eval on a held slice -> should jump >> chance.

Run:  python dino/training_scripts/confirm_dino.py [--input_size 224]
"""
import sys, os, argparse
sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

ap = argparse.ArgumentParser()
ap.add_argument("--input_size", type=int, default=224)
args, _ = ap.parse_known_args()
INPUT = args.input_size
assert INPUT % 14 == 0, f"--input_size must be divisible by 14 (ViT-S/14); got {INPUT}"

NUM_CLASSES, EMBED_DIM = 13, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
print(f"Device: {DEVICE}  input_size: {INPUT}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class DinoClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


def build_model(input_size):
    load_path = None
    backbone = None
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        load_path = "hub:dinov2_vits14"
    except Exception as e:
        print(f"[load] torch.hub failed ({type(e).__name__}: {e}); falling back to timm.")
        import timm
        backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                     pretrained=True, num_classes=0, img_size=input_size)
        load_path = "timm:vit_small_patch14_dinov2.lvd142m"
    return DinoClassifier(backbone), load_path


def prep(x, input_size):
    x = x.to(DEVICE)
    x = T.Resize((input_size, input_size), antialias=True)(x)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


print("\n[1] Loading DINOv2 ViT-S/14 ...")
model, LOAD_PATH = build_model(INPUT)
model = model.to(DEVICE).eval()
print(f"  loaded via: {LOAD_PATH}")
print(f"  head: Linear({model.head.in_features}, {model.head.out_features})")
assert model.head.in_features == EMBED_DIM and model.head.out_features == NUM_CLASSES

print("\n[2] Forward on a 100x100 crop batch (resize -> normalize) ...")
x = torch.rand(8, 3, 100, 100)
with torch.no_grad():
    logits = model(prep(x, INPUT))
print(f"  100x100 -> resize {INPUT} -> logits {tuple(logits.shape)}")
assert logits.shape == (8, NUM_CLASSES) and torch.isfinite(logits).all()
print("  finite (B, 13) ✓")

print("\n[3] Token count (patch tokens = (INPUT/14)^2) ...")
for sz in (224, 98, INPUT):
    n = (sz // 14) ** 2
    print(f"  input {sz:>3} -> {sz//14}x{sz//14} = {n} patch tokens (+1 CLS)")
# empirical confirmation via forward_features if available
try:
    with torch.no_grad():
        ff = model.backbone.forward_features(prep(torch.rand(1, 3, 100, 100), INPUT))
    if isinstance(ff, dict) and "x_norm_patchtokens" in ff:
        print(f"  empirical (hub forward_features x_norm_patchtokens): {ff['x_norm_patchtokens'].shape[1]} tokens @ {INPUT}")
    elif torch.is_tensor(ff):
        print(f"  empirical (timm forward_features): {tuple(ff.shape)} (incl CLS/registers)")
except Exception as e:
    print(f"  (empirical token introspection skipped: {type(e).__name__})")

print("\n[4] Param count ...")
total = sum(p.numel() for p in model.parameters())
bb = sum(p.numel() for p in model.backbone.parameters())
hd = sum(p.numel() for p in model.head.parameters())
print(f"  DINOv2 ViT-S/14 + 13-class head: {total:,} params (backbone={bb:,}, head={hd:,})")
print(f"  reference: ResNet-18 11,183,181 | ConvNeXt-Tiny 27,830,125")

print("\n[5] Norm layers ...")
bn = sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())
ln = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
print(f"  BatchNorm={bn} (expect 0)   LayerNorm={ln}")
assert bn == 0, "unexpected BatchNorm in DINOv2"

print("\n[6] Write-guard check (frozen-token run_name must abort before write) ...")
EXP_DIR = "/home/eladbaum/chess_project/training/dino"
_FROZEN = ("zero_shot", "stage1_10", "stage2_30", "stage3_323", "stage3_improved",
           "stage5_combined_323", "convnext")
def guard(run_name):
    d = os.path.realpath(f"{PROJECT_ROOT}/checkpoints/{run_name}")
    assert d.startswith(os.path.realpath(EXP_DIR) + os.sep), "not under dino/"
    for tok in _FROZEN:
        assert tok not in d, f"names frozen baseline '{tok}'"
guard("dino_zeroshot")  # legit — should pass
print("  legit run_name 'dino_zeroshot' passes guard ✓")
for bad in ("convnext_stage5", "stage3_323"):
    try:
        guard(bad); raise SystemExit(f"GUARD FAILED to block '{bad}'")
    except AssertionError as e:
        print(f"  guard correctly aborts run_name '{bad}': {e}")

print("\n[7] Verbatim eval-harness imports ...")
from rescan_checkpoint_selection import RealGameDataset, metrics  # noqa: F401
print("  imported RealGameDataset + metrics from rescan_checkpoint_selection ✓")

print("\n[8] DINO sanity linear-probe (frozen features -> head fit on real squares) ...")
PROJECT_ROOT = "/home/eladbaum/chess_project"
ds = RealGameDataset(f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv",
                     f"{PROJECT_ROOT}/data/game7_per_frame/images", "game7", transform=None)
rng = np.random.RandomState(42)
idxs = rng.choice(len(ds), size=min(800, len(ds)), replace=False)
feats, labs = [], []
with torch.no_grad():
    for i in range(0, len(idxs), 64):
        batch = [ds[int(j)] for j in idxs[i:i + 64]]
        xb = torch.stack([b[0] for b in batch])
        yb = np.array([b[1] for b in batch])
        f = model.backbone(prep(xb, INPUT))
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.cpu().numpy()); labs.append(yb)
X = np.concatenate(feats); Y = np.concatenate(labs)
ntr = int(0.7 * len(Y))
perm = rng.permutation(len(Y)); tr, te = perm[:ntr], perm[ntr:]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X[tr], Y[tr])
acc = float((clf.predict(X[te]) == Y[te]).mean())
# baselines on the test slice
vals, cnts = np.unique(Y[te], return_counts=True)
majority = float(cnts.max() / cnts.sum())
print(f"  {len(Y)} game7 squares, {X.shape[1]}-d frozen CLS features; LogReg test acc on {len(te)} held = {acc:.4f}")
print(f"  baselines: uniform-chance={1/NUM_CLASSES:.4f}  majority-class(empty)={majority:.4f}")
assert acc > majority + 0.05, (
    f"linear-probe acc {acc:.4f} not clearly above majority {majority:.4f} — resize/normalize path "
    f"may not be feeding DINO valid features.")
print("  linear-probe jumps above majority baseline -> resize/normalize path feeds valid DINO features ✓")

print("\n\033[92m✓ DINOv2 ViT-S/14 confirmed: loads, head Linear(384,13), runs on 100x100->"
      f"{INPUT}, ~{total/1e6:.0f}M params, BN=0, guard fires, eval imports OK, frozen-feature "
      "linear-probe >> chance.\033[0m")
