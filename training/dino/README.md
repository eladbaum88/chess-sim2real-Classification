# DINOv2 ViT-S/14 — training

Training code for the **DINOv2 ViT-S/14** backbone (self-supervised ViT, ~21M params), our
strongest model for sim-to-real transfer. It runs under the same data, splits, and
warp/crop pipeline as the **ResNet-18** (11.2M) and **ConvNeXt-Tiny** (27.8M) comparison
backbones, so the numbers are directly comparable.

## Runs

| Run | Style | Initialisation | Data | Sampler | Selection |
|---|---|---|---|---|---|
| `dino_zeroshot` | synthetic-only | DINOv2 pretrained | full `dataset_v1` synth | shuffle (90/10 by image) | synth val |
| `dino_fine_tuned` | sequential fine-tune | `dino_zeroshot/best_synth.pt` | 30 manual + game4 + game5 | shuffle | real val |
| `dino_combined` | combined | DINOv2 pretrained | synth + 30 manual + game4 + game5 | weighted 50/50, 100k/epoch | real val |
| `dino_combined_linprob` | linear probe | DINOv2 pretrained | combined | weighted 50/50, 100k/epoch | real val |

`dino_combined_linprob` keeps the backbone **frozen** throughout (head only) — the canonical
DINOv2 usage, and the cleanest read on the frozen features.

## Recipe

- **Model:** DINOv2 ViT-S/14 → CLS embedding (384) → `Linear(384, 13)`. Loaded from the
  vendored code (`torch.hub`/`timm` fallback during training).
- **Input:** datasets yield 100×100 crops; a `Resize((224, 224), antialias=True)` is applied
  at the model boundary, just before ImageNet normalisation (224 = 16×16 = 256 patch tokens;
  must be divisible by 14 for ViT-S/14).
- **Two-phase fine-tuning:** Phase A freezes the backbone and trains the head; Phase B
  unfreezes everything with discriminative learning rates (head `1e-4`, backbone `1e-5` — ViT
  fine-tuning is more delicate than a CNN's). The linear probe skips Phase B.
- **Optimiser:** AdamW + cosine schedule, weight decay 0.05, batch 64, seed 42. Each run's
  `recipe.json` records the exact configuration.

## Run order

```bash
PY=~/.conda/envs/chess/bin/python
cd training/dino

$PY -u train.py --mode zeroshot --run_name dino_zeroshot
$PY -u train.py --mode fine_tuned   --run_name dino_fine_tuned
$PY -u train.py --mode combined   --run_name dino_combined
$PY -u train.py --mode linprobe --run_name dino_combined_linprob
# ablations (dedicated scripts):
$PY -u train_combindedGame6_diag.py  --run_name dino_combined_Game6boosted
$PY -u train_realonly_ablation.py    --run_name dino_realOnly
$PY -u train_labelsmooth_ablation.py --run_name dino_combined_Game6boosted_ablation_LabelSmoothing
```

Checkpoints are written per run under `checkpoints/<run>/`; only each run's `best_real.pt` is
committed (see [`../../checkpoints/README.md`](../../checkpoints/README.md)).

> **Note.** These scripts depend on the shared eval module `rescan_checkpoint_selection.py`
> (under `training/resnet18/`) and the helpers `eval_games_2_6.py` / `build_report.py` /
> `confirm_dino.py`, all kept local-only. Training therefore runs from the full local/Drive
> tree, not a bare clone — the committed inference path (`evaluation/`) is fully self-sufficient.
