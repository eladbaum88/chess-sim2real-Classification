"""Build realonly_vs_combined_compare.json — real-only ablation vs dino_combined_Game6boosted on game7.

Both sides are the game2-SELECTED best checkpoint evaluated on the held-out game7 (apples-to-apples).
Headline metric is MACRO-AVERAGE (mean of 13 per-class accs) + per-class deltas on the rare/tall
pieces; per-square is reported but flagged as empty-dominated/misleading.
"""
import json, numpy as np

NAMES = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]
REALONLY = "/home/eladbaum/chess_project/training/dino/dino_realOnly/results/predictions"
COMBINED = "/home/eladbaum/chess_project/training/dino/results/dino_combined_Game6boosted/predictions"
OUT = "/home/eladbaum/chess_project/training/dino/dino_realOnly/results/realonly_vs_combined_compare.json"
HIGHLIGHT = {"wB": 3, "bB": 9, "wQ": 4, "bQ": 10, "wK": 5, "bK": 11}  # bishops + royalty


def metrics(preds, labels):
    per_class = {}
    for ci, nm in enumerate(NAMES):
        m = labels == ci
        per_class[nm] = float((preds[m] == ci).mean()) if m.any() else None
    vals = [v for v in per_class.values() if v is not None]
    macro = float(np.mean(vals))
    piece_m = labels != 12
    piece_only = float((preds[piece_m] == labels[piece_m]).mean())
    per_square = float((preds == labels).mean())
    return {"macro_average": macro, "piece_only": piece_only, "per_square": per_square,
            "per_class": per_class}


def load(d):
    return np.load(f"{d}/game7_preds.npy"), np.load(f"{d}/game7_labels.npy")


ro_p, ro_y = load(REALONLY)
cb_p, cb_y = load(COMBINED)
assert np.array_equal(ro_y, cb_y), "game7 label arrays differ between runs — not comparable!"

ro, cb = metrics(ro_p, ro_y), metrics(cb_p, cb_y)
deltas = {nm: (None if ro["per_class"][nm] is None or cb["per_class"][nm] is None
               else round(ro["per_class"][nm] - cb["per_class"][nm], 4)) for nm in NAMES}

out = {
    "test": "game7 (held out); both = game2-selected best checkpoint, eval-fixed",
    "ablation": "remove synthetic half (combined -> real-only); single variable",
    "headline_metric": "macro_average (per-square is empty-dominated and misleading)",
    "real_only": ro,
    "combined_game6_baseline": cb,
    "delta_realonly_minus_combined": {
        "macro_average": round(ro["macro_average"] - cb["macro_average"], 4),
        "piece_only": round(ro["piece_only"] - cb["piece_only"], 4),
        "per_square": round(ro["per_square"] - cb["per_square"], 4),
        "per_class": deltas,
    },
    "highlight_rare_tall_pieces": {nm: {"combined": cb["per_class"][nm],
                                        "real_only": ro["per_class"][nm],
                                        "delta": deltas[nm]} for nm in HIGHLIGHT},
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

print("=== REAL-ONLY vs COMBINED (game7, game2-selected) ===")
print(f"  macro-avg : real-only {ro['macro_average']:.4f}  vs combined {cb['macro_average']:.4f}  "
      f"(Δ {out['delta_realonly_minus_combined']['macro_average']:+.4f})")
print(f"  piece-only: real-only {ro['piece_only']:.4f}  vs combined {cb['piece_only']:.4f}  "
      f"(Δ {out['delta_realonly_minus_combined']['piece_only']:+.4f})")
print(f"  per-square: real-only {ro['per_square']:.4f}  vs combined {cb['per_square']:.4f}  "
      f"(Δ {out['delta_realonly_minus_combined']['per_square']:+.4f})  [misleading: ~55% empty]")
print("  rare/tall pieces (combined -> real-only, Δ):")
for nm in HIGHLIGHT:
    print(f"    {nm}: {cb['per_class'][nm]:.3f} -> {ro['per_class'][nm]:.3f}  ({deltas[nm]:+.3f})")
print(f"\nwrote {OUT}")
