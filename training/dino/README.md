# DINOv2 ViT-S/14 — third backbone

Adds **DINOv2 ViT-S/14** (self-supervised ViT, ~21M params) to the sim-to-real comparison
alongside **ResNet-18** (11.2M) and **ConvNeXt-Tiny** (27.8M). Mirrors `convnext/` exactly —
same data, splits, crop/warp pipeline, 5% forgetting probe, game7 monitor, and the verbatim
games-2/6 eval — so numbers are directly comparable. Only the documented DINO changes differ.

## Runs

| run | style | source | data | sampler | selection |
|---|---|---|---|---|---|
| `dino_zeroshot` | synth-only | DINOv2 pretrained | full dataset_v1 synth | shuffle (90/10 by-image) | synth val |
| `dino_fine_tuned` | sequential FT | `dino_zeroshot/best_synth.pt` | 30 manual + game4 + game5 | shuffle | game7 |
| `dino_combined` | combined | DINOv2 pretrained | synth + 30 manual + game4 + game5 | Weighted 50/50, 100k/epoch | game7 |
| `dino_combined_linprob` | linear probe | DINOv2 pretrained | combined (same as stage5) | Weighted 50/50, 100k/epoch | game7 |

`dino_combined_linprob` keeps the backbone **frozen for all epochs** (head only) — the canonical
DINOv2 usage and the cleanest DINO result; the CNNs have no equivalent.

## What differs from convnext (everything else identical)

- **Model:** DINOv2 ViT-S/14 (`torch.hub` `dinov2_vits14`, timm
  `vit_small_patch14_dinov2.lvd142m` fallback) → CLS embedding (384) → `Linear(384, 13)`.
- **Input resize:** datasets still yield 100×100 crops (byte-identical to ResNet/ConvNeXt).
  `transforms.Resize((INPUT, INPUT), antialias=True)` is applied at the model boundary,
  immediately before ImageNet-normalize, in train + eval + games-2/6. `--input_size` default
  **224** (16×16 = **256 patch tokens**); must be divisible by 14 (ViT-S/14). 98 (49 tokens)
  is a later sensitivity option. *(Resize-at-boundary == resize-in-transform-stack: bilinear
  resize commutes with the per-channel affine normalize.)*
- **Freeze (ViT two-phase):** Phase A freezes the whole backbone (patch_embed, 12 blocks,
  norm, cls_token, pos_embed), trains the head; Phase B unfreezes all with discriminative LRs
  (head 1e-4, backbone **1e-5** — ViT FT is more fragile than ConvNeXt's 3e-5). LayerNorm only
  → BN-freeze N/A. `linprobe` skips Phase B entirely.
- **AdamW + cosine, wd 0.05, batch 64, seed 42** (same as convnext). Per-run `recipe.json`
  records the load path, token count, freeze scheme, LRs, and the resize note.

## Write discipline

Outputs route through `--run_name` under `dino/`. Hard write-guard (in `train.py` and
`eval_games_2_6.py`) asserts every path resolves under `dino/` and names no frozen baseline:
`zero_shot, stage1_10, stage2_30, stage3_323, stage3_improved, stage5_combined_323, convnext`.

## Run order

```bash
PY=~/.conda/envs/chess/bin/python
cd /home/eladbaum/chess_project/training/dino

$PY -u train.py --mode zeroshot --run_name dino_zeroshot
$PY -u train.py --mode stage3   --run_name dino_fine_tuned
$PY -u train.py --mode stage5   --run_name dino_combined
$PY -u train.py --mode linprobe --run_name dino_combined_linprob
# special runs (dedicated scripts):
$PY -u train_combindedGame6_diag.py --run_name dino_combined_Game6boosted
$PY -u train_realonly_ablation.py   --run_name dino_realOnly
$PY -u train_labelsmooth_ablation.py --run_name dino_combined_Game6boosted_ablation_LabelSmoothing
```

`eval_games_2_6.py --run_name <run>` re-runs the verbatim games-2/6 eval (with the DINO
resize) on any run's best checkpoint.
