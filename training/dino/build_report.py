"""Aggregate the DINOv2 runs into the architecture-comparison table (md + csv), alongside the
published ResNet-18 and ConvNeXt-Tiny numbers. Reads each dino run's games_2_6_eval.json,
synth_monitor_results.json, recipe.json. Same output format as convnext/build_report.py.

Usage:  python build_report.py
"""
import sys, os, json
sys.path.insert(0, "/home/eladbaum/chess_project")
import csv as _csv

EXP_DIR = "/home/eladbaum/chess_project/training/dino"
RESULTS = f"{EXP_DIR}/results"

# Published reference numbers (games 2/6) — hard-coded.
RESNET = {"zeroshot": (0.5138, None), "stage3": (0.9085, 0.7556), "stage5": (0.9160, 0.7748)}
CONVNEXT = {"zeroshot": (0.7960, 0.4621), "stage3": (0.9468, 0.8589), "stage5": (0.9557, 0.8828)}
# rows: (label, dino_run, regime_key_for_refs or None for linprobe)
ROWS = [("synth-only (zero-shot)", "dino_zeroshot", "zeroshot"),
        ("real fine-tune (Stage 3)", "dino_fine_tuned", "stage3"),
        ("combined (Stage 5)", "dino_combined", "stage5"),
        ("linear-probe (frozen DINO)", "dino_combined_linprob", None)]
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]


def _load(run, name):
    p = f"{RESULTS}/{run}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


lines = ["# DINOv2 ViT-S/14 vs ConvNeXt-Tiny vs ResNet-18 — architecture comparison (games 2/6)\n"]
lines.append("| model (games 2/6) | ResNet per-sq | ConvNeXt per-sq | DINOv2 per-sq | "
             "ResNet piece | ConvNeXt piece | DINOv2 piece | DINOv2 forget Δ |")
lines.append("|---|---|---|---|---|---|---|---|")
table_csv = [["row", "resnet_per_sq", "convnext_per_sq", "dino_per_sq",
              "resnet_piece", "convnext_piece", "dino_piece", "dino_forget_delta",
              "dino_game7_per_sq", "selected_epoch"]]

for label, run, key in ROWS:
    g26 = _load(run, "games_2_6_eval.json")
    forget = _load(run, "synth_monitor_results.json")
    recipe = _load(run, "recipe.json")
    d_persq = g26["per_square_acc"] if g26 else None
    d_piece = g26["piece_only_acc"] if g26 else None
    delta = forget["forgetting_delta"] if forget else None
    g7 = recipe["results"]["game7_per_square"] if recipe else None
    sel = recipe.get("selected_epoch") if recipe else None
    r_persq, r_piece = RESNET.get(key, (None, None))
    c_persq, c_piece = CONVNEXT.get(key, (None, None))
    lines.append(f"| {label} | {f(r_persq)} | {f(c_persq)} | {f(d_persq)} | "
                 f"{f(r_piece)} | {f(c_piece)} | {f(d_piece)} | "
                 f"{f(delta) if delta is not None else '—'} |")
    table_csv.append([label, r_persq, c_persq, d_persq, r_piece, c_piece, d_piece, delta, g7, sel])

lines.append("\n## Per-class held-out (games 2/6) accuracy — DINOv2\n")
lines.append("| run | " + " | ".join(CLASS_SHORT) + " |")
lines.append("|---|" + "---|" * len(CLASS_SHORT))
for label, run, key in ROWS:
    g26 = _load(run, "games_2_6_eval.json")
    if not g26:
        continue
    pc = g26["per_class_acc"]
    lines.append(f"| {label} | " + " | ".join(
        (f"{pc[c]:.3f}" if pc.get(c) is not None else "—") for c in CLASS_SHORT) + " |")

lines.append("\n## Recipe (each architecture done right)\n")
lines.append("ResNet-18: SGD + two-phase freeze. ConvNeXt-Tiny (~27.8M): AdamW + cosine + ConvNeXt "
             "two-phase freeze. DINOv2 ViT-S/14 (~21M): AdamW + cosine, ViT two-phase freeze, inputs "
             "resized 100->224 (256 patch tokens) before ImageNet-normalize, backbone LR 1e-5 (ViT FT "
             "is fragile). LayerNorm only -> BN-freeze N/A. linprobe = frozen backbone, head only.")
for label, run, key in ROWS:
    recipe = _load(run, "recipe.json")
    if recipe:
        lines.append(f"\n- **{label}** (`{run}`): {recipe['optimizer']}, lr_head={recipe['lr_head']}, "
                     f"lr_backbone={recipe['lr_backbone']}, input={recipe['input_resolution']} "
                     f"({recipe['patch_tokens']} tokens), epochs={recipe['epochs']}, select "
                     f"{recipe['selection_metric']} @ ep {recipe.get('selected_epoch')}, "
                     f"load={recipe.get('dino_load_path')}.")

md = "\n".join(lines) + "\n"
open(f"{RESULTS}/comparison_report.md", "w").write(md)
with open(f"{RESULTS}/comparison_report.csv", "w", newline="") as fh:
    _csv.writer(fh).writerows(table_csv)
print(md)
print(f"wrote {RESULTS}/comparison_report.md + .csv")
