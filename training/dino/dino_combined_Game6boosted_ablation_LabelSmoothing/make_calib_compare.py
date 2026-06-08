"""Label-smoothing vs combined_game6 on game7 — accuracy AND calibration.

Both sides = the game2-SELECTED best_real.pt, evaluated on the held-out game7 (eval-fixed). We load
each checkpoint and run game7 inference capturing SOFTMAX PROBABILITIES (the saved .npy hold argmax
only), then report accuracy {macro, piece-only, per-square, per-class + tall deltas} and calibration
{ECE-15bin, mean confidence on errors / correct, tall-piece confidence-on-errors}.
"""
import json, csv, numpy as np, cv2, torch, torch.nn as nn
import torchvision.transforms as T
import sys; sys.path.insert(0, "/home/eladbaum/chess_project")
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (warp_chessboard_image, crop_square, find_corners,
                                            ChessboardNotLocatedException)

NAMES = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]
TALL = {"wB": 3, "bB": 9, "wQ": 4, "bQ": 10, "wK": 5, "bK": 11}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
RESIZE = T.Resize((224, 224), antialias=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)
ROOT = "/home/eladbaum/chess_project"
CKPTS = {"combined_game6": f"{ROOT}/dino/checkpoints/dino_combined_Game6boosted/best_real.pt",
         "label_smoothing": f"{ROOT}/dino/dino_combined_Game6boosted_ablation_LabelSmoothing/checkpoints/best_real.pt"}
OUT = f"{ROOT}/dino/dino_combined_Game6boosted_ablation_LabelSmoothing/results/labelsmooth_vs_combined_compare.json"


class Dino(nn.Module):
    def __init__(s, b): super().__init__(); s.backbone = b; s.head = nn.Linear(384, 13)
    def forward(s, x):
        f = s.backbone(x); f = f[0] if isinstance(f, (tuple, list)) else f; return s.head(f)


def build_game7_crops():
    rows = list(csv.DictReader(open(f"{ROOT}/data/game7_per_frame/gt.csv")))
    crops, labels = [], []
    for r in rows:
        bgr = cv2.imread(f"{ROOT}/data/game7_per_frame/images/{r['image_name']}")
        H, W = bgr.shape[:2]
        try:
            np.random.seed(SEED); c = find_corners(bgr)
            if not bool(np.all((c[:, 0] >= -8) & (c[:, 0] <= W + 8) & (c[:, 1] >= -8) & (c[:, 1] <= H + 8))):
                raise ChessboardNotLocatedException()
        except Exception:
            c = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        warped = warp_chessboard_image(bgr, c); grid = fen_to_label_grid(r["fen"], "game7")
        for br in range(8):
            for bc in range(8):
                crops.append(cv2.cvtColor(crop_square(warped, br, bc), cv2.COLOR_BGR2RGB))
                labels.append(int(grid[br, bc]))
    return crops, np.array(labels)


@torch.no_grad()
def probs_for(ckpt_path, crops):
    bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    m = Dino(bb).to(DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True); m.eval()
    out = np.zeros((len(crops), 13), dtype=np.float32)
    B = 256
    for i in range(0, len(crops), B):
        x = torch.from_numpy(np.stack(crops[i:i + B])).permute(0, 3, 1, 2).float().to(DEV) / 255.0
        x = (RESIZE(x) - MEAN) / STD
        out[i:i + B] = torch.softmax(m(x), 1).cpu().numpy()
    return out, int(ck.get("epoch", -1))


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def metrics(probs, labels):
    preds = probs.argmax(1); conf = probs.max(1); correct = (preds == labels).astype(float)
    per_class = {nm: (float((preds[labels == ci] == ci).mean()) if (labels == ci).any() else None)
                 for ci, nm in enumerate(NAMES)}
    vals = [v for v in per_class.values() if v is not None]
    pm = labels != 12
    tall_mask = np.isin(labels, list(TALL.values()))
    err = preds != labels
    return {
        "macro_average": float(np.mean(vals)),
        "piece_only": float((preds[pm] == labels[pm]).mean()),
        "per_square": float(correct.mean()),
        "per_class": per_class,
        "ece_15bin": ece(conf, correct),
        "mean_conf_on_errors": float(conf[err].mean()) if err.any() else None,
        "mean_conf_on_correct": float(conf[~err].mean()),
        "tall_mean_conf_on_errors": float(conf[tall_mask & err].mean()) if (tall_mask & err).any() else None,
    }


crops, labels = build_game7_crops()
res = {}
for name, path in CKPTS.items():
    p, ep = probs_for(path, crops)
    res[name] = metrics(p, labels); res[name]["epoch"] = ep

cb, ls = res["combined_game6"], res["label_smoothing"]
deltas_pc = {nm: (None if cb["per_class"][nm] is None or ls["per_class"][nm] is None
                  else round(ls["per_class"][nm] - cb["per_class"][nm], 4)) for nm in NAMES}
out = {
    "test": "game7 (held out); both = game2-selected best_real.pt, eval-fixed",
    "ablation": "label smoothing (train-CE, 0.1) vs combined_game6 (plain CE); single variable",
    "combined_game6": cb, "label_smoothing": ls,
    "delta_ls_minus_combined": {
        "macro_average": round(ls["macro_average"] - cb["macro_average"], 4),
        "piece_only": round(ls["piece_only"] - cb["piece_only"], 4),
        "per_square": round(ls["per_square"] - cb["per_square"], 4),
        "ece_15bin": round(ls["ece_15bin"] - cb["ece_15bin"], 4),
        "mean_conf_on_errors": round(ls["mean_conf_on_errors"] - cb["mean_conf_on_errors"], 4),
        "tall_mean_conf_on_errors": round(ls["tall_mean_conf_on_errors"] - cb["tall_mean_conf_on_errors"], 4),
        "per_class": deltas_pc,
    },
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

d = out["delta_ls_minus_combined"]
print("=== LABEL SMOOTHING vs COMBINED (game7, game2-selected) ===")
print(f"  ACCURACY  macro-avg {cb['macro_average']:.4f} -> {ls['macro_average']:.4f} (Δ {d['macro_average']:+.4f})")
print(f"            piece-only {cb['piece_only']:.4f} -> {ls['piece_only']:.4f} (Δ {d['piece_only']:+.4f})")
print(f"            per-square {cb['per_square']:.4f} -> {ls['per_square']:.4f} (Δ {d['per_square']:+.4f}) [empty-dominated]")
print(f"  CALIB     ECE-15bin {cb['ece_15bin']:.4f} -> {ls['ece_15bin']:.4f} (Δ {d['ece_15bin']:+.4f})  [lower=better]")
print(f"            conf-on-errors {cb['mean_conf_on_errors']:.4f} -> {ls['mean_conf_on_errors']:.4f} (Δ {d['mean_conf_on_errors']:+.4f})  [lower=better]")
print(f"            conf-on-correct {cb['mean_conf_on_correct']:.4f} -> {ls['mean_conf_on_correct']:.4f}")
print(f"            tall-piece conf-on-errors {cb['tall_mean_conf_on_errors']:.4f} -> {ls['tall_mean_conf_on_errors']:.4f} (Δ {d['tall_mean_conf_on_errors']:+.4f})")
print("  tall-piece accuracy Δ (LS - combined):")
for nm in TALL:
    print(f"    {nm}: {cb['per_class'][nm]:.3f} -> {ls['per_class'][nm]:.3f} ({deltas_pc[nm]:+.3f})")
print(f"\nwrote {OUT}")
