# Run 2 — `dino_fine_tuned` (DINOv2 ViT-S/14, sequential fine-tune synth→real)

Sequential fine-tune: starts from the `dino_zeroshot` synth checkpoint (epoch 10, synth_val
0.9999) and trains on real data only (30 manual + game4 + game5 PGN). Mirrors
`convnext_stage3` / ResNet Stage 3. This is the FT arm of the architecture comparison.

## Setup

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14 (`hub:dinov2_vits14`), 22.06M params |
| Source weights | `dino_zeroshot/best_synth.pt` (epoch 10, synth_val 0.9999) — NOT ImageNet/random |
| Head | `Linear(384, 13)` on CLS embedding |
| Input | 100×100 crops → resize **224** (256 tokens) before ImageNet-normalize |
| Recipe | AdamW, wd 0.05, cosine, batch 64, seed 42; head lr 1e-4, **backbone lr 1e-5** |
| Freeze | Phase A (ep1–2, head-only) → Phase B (ep3+, all unfrozen) |
| Data | real only: 30 manual (games 8-11) + game4 (184) + game5 (109) = 323 frames / ~20,672 squares |
| Selection | **game7 real_val**, early-stop patience 6 |
| Hardware | RTX 3090; **33.8 min**, 13 epochs (early-stopped), ~2.4 min/epoch |

## Leakage / data verification (pre-launch)

- Forgetting-probe synth slice = `dataset_v1` (NOT v1.5).
- Real train = 30 manual (games {8,9,10,11}) + game4 + game5 = 323 frames; asserts held.
- Train games {4,5,8,9,10,11} disjoint from eval games {7,2,6}; real-train ∩ (game7 ∪ games-2/6) = 0.
- Pre-FT smoke: synth_monitor **0.9999** (>0.95 assert → confirms synth-trained weights loaded),
  game7 before 0.6969 (matches the zeroshot checkpoint — correct source).

## Training trajectory (selection = game7)

| epoch | phase | train | game7 (sel) | synth_monitor (forgetting) |
|---|---|---|---|---|
| 1 | A | 0.8375 | 0.8000 | 0.9990 |
| 2 | A | 0.8951 | 0.8293 | 0.9966 |
| 3 | B | 0.9737 | 0.9580 | 0.9847 |
| 4 | B | 0.9907 | 0.9565 | 0.9813 |
| 5 | B | 0.9935 | 0.9645 | 0.9766 |
| 6 | B | 0.9950 | 0.9580 | 0.9714 |
| **7** | **B** | 0.9960 | **0.9795** | 0.9785 |
| 8–13 | B | →0.999 | 0.972–0.976 | →0.966 |

- **Selected epoch 7** → `checkpoints/dino_fine_tuned/best_real.pt`. Early-stopped at epoch 13
  (game7 no improvement over 0.9795 for 6 epochs).
- **No underfit, no over-adaptation:** Phase-B onset (ep3) jumped train 0.895→0.974 and game7
  0.829→0.958 — the backbone adapts strongly even at the cautious **1e-5**, so no 3e-5 rerun is
  needed (the ViT-fragility caution did not materialize on this small real set). game7 plateaued
  ~0.97–0.98 thereafter; no collapse.

## Held-out games 2/6 — DINO is the best of all three, both axes

| metric | DINOv2 | ConvNeXt-Tiny | ResNet-18 |
|---|---|---|---|
| **per-square** | **0.9588** | 0.9468 | 0.9085 |
| **piece-only** | **0.8908** | 0.8589 | 0.7556 |
| empty | 0.9988 | 0.9987 | — |
| game2 (per-sq / piece) | 0.9909 / 0.9766 | — | — |
| game6 (per-sq / piece) | 0.9319 / 0.8244 | — | — |

FT helped DINO enormously vs its own zero-shot (0.780/0.548 → **0.959/0.891**; piece-only +0.34).
game6 remains the hard game (the shared far/near-rank warp ceiling seen across all architectures).

## True-forgetting Δ — third architecture corroborating the BN-stat story

**Δ = −0.0214** (synth_monitor 0.9999 → 0.9785 at the epoch-7 checkpoint).

| arch | stage3 forgetting Δ | normalization |
|---|---|---|
| ResNet-18 | −0.13 | BatchNorm |
| ConvNeXt-Tiny | −0.036 | LayerNorm |
| **DINOv2 ViT-S/14** | **−0.0214** | LayerNorm |

Both LayerNorm-only backbones forget far less than the BatchNorm ResNet; DINO forgets even less
than ConvNeXt. A **third** architecture consistent with BN running-stat drift as the forgetting
mechanism — no BatchNorm → minimal catastrophic forgetting.

## Anchor-class confusion check — zeroshot collapse FIXED by real data alone

The classes that collapsed near-zero in zero-shot (tall pieces → wB/bQ "anchor" columns) recover
sharply after real fine-tuning:

| class | zeroshot | stage3 | Δ |
|---|---|---|---|
| wK | 0.071 | 0.899 | +0.83 |
| bK | 0.000 | 0.846 | +0.85 |
| wN | 0.006 | 0.781 | +0.78 |
| wQ | 0.205 | 0.644 | +0.44 |
| wP/wR/bP/bR | 0.60–0.78 | 0.97–0.99 | + |

→ The king/knight confusion is a **synth-appearance gap repaired by real crops**, NOT a feature-
separability problem → **no contrastive-loss ablation warranted.** Caveat — two regressions: **bB
0.617→0.457, bQ 0.519→0.366** (real train set is thin on black bishops/queens; the model trades
some of those for the large king/knight gains). wB stays weak (0.443).

## Artifacts

`checkpoints/dino_fine_tuned/{best_real.pt(ep7), best_synth_monitor.pt, latest.pt}`;
`results/dino_fine_tuned/{training_log.csv, recipe.json, games_2_6_eval.json, game7_results.json,
synth_monitor_results.json, predictions/*.npy}`; `plots/dino_fine_tuned/*.png`.

## Next

`dino_combined` — combined synth+real from DINOv2-pretrained, WeightedRandomSampler 50/50
(100k/epoch), backbone 1e-5, select game7, patience-6. Forgetting Δ there is acquisition vs
DINOv2 (not true forgetting); the meaningful retention number is vs `dino_zeroshot`. **A
summary.md will be created for run 3 as well.**
