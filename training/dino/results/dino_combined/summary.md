# Run 3 — `dino_combined` (DINOv2 ViT-S/14, combined synth+real from pretrained)

Joint/combined training: starts from DINOv2-pretrained (NOT the synth checkpoint, same as
ConvNeXt stage5 started from ImageNet) and trains on synth + real combined via a 50/50
WeightedRandomSampler. Mirrors `convnext_stage5` / ResNet Stage 5.

## Setup

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14 (`hub:dinov2_vits14`), 22.06M params |
| Source | DINOv2-pretrained + fresh head (NOT dino_zeroshot) |
| Input | 100×100 crops → resize **224** (256 tokens) before ImageNet-normalize |
| Recipe | AdamW, wd 0.05, cosine, batch 64, seed 42; head lr 1e-4, **backbone lr 1e-5** |
| Freeze | Phase A (ep1–2, head-only) → Phase B (ep3+, all unfrozen) |
| Data | combined: synth (392,448 sq) + real (20,672 sq); WeightedRandomSampler 50/50, 100k draws/epoch |
| Selection | **game7 real_val**, patience 6 |
| Hardware | RTX 3090; **139.2 min (~2.3h)**, 20 epochs (ran full), ~7 min/epoch |

## Leakage / data verification (pre-launch)

- Synth = `dataset_v1` (NOT v1.5); stage5 trains on synth (the 50/50 synth half). Combined =
  synth 392,448 + real 20,672 squares (natural 19:1, rebalanced to 50/50 by the sampler).
- Real half = 30 manual (games {8,9,10,11}) + game4 + game5; asserts held.
- Combined-train ∩ (game7 ∪ games-2/6) = 0 (game-qualified). Selection = game7.
- Pre-FT smoke: synth_monitor before **0.6473**, game7 before 0.0622 — identical to dino_zeroshot
  pre-train → confirms fresh DINOv2-pretrained start (the `<0.85` assert correctly fired; a ~0.99
  would mean a chess-trained checkpoint loaded by mistake).

## Training trajectory (selection = game7)

| epoch | phase | train | game7 (sel) | synth_monitor |
|---|---|---|---|---|
| 1 | A | 0.7871 | 0.6611 | 0.7827 |
| 2 | A | 0.8485 | 0.7011 | 0.8141 |
| 3 | B | 0.9729 | 0.9509 | 0.9789 |
| 4 | B | 0.9892 | 0.9631 | 0.9901 |
| 7 | B | 0.9948 | 0.9795 | 0.9958 |
| 8 | B | 0.9950 | 0.9798 | 0.9970 |
| 13 | B | 0.9979 | 0.9832 | 0.9993 |
| 15 | B | 0.9992 | 0.9858 | 0.9995 |
| **19** | **B** | 0.9994 | **0.9872** | 0.9998 |
| 20 | B | 0.9996 | 0.9866 | 0.9997 |

- **Selected epoch 19** → `checkpoints/dino_combined/best_real.pt`. Ran the full 20 epochs (kept
  finding small new game7 bests, so patience-6 never tripped).
- **No underfit / no over-adaptation:** Phase-B onset (ep3) jumped train 0.849→0.973 and game7
  0.701→0.951 — clean strong adaptation at 1e-5, same as stage3. synth_monitor climbed to ~0.9998
  (combined trains on synth, so synth ability is acquired/retained, not lost).

## Held-out games 2/6 — DINO sweeps every regime; combined > sequential confirmed for all 3 archs

| metric | DINOv2 stage5 | DINOv2 stage3 | ConvNeXt stage5 | ResNet stage5 |
|---|---|---|---|---|
| **per-square** | **0.9761** | 0.9588 | 0.9557 | 0.9160 |
| **piece-only** | **0.9377** | 0.8908 | 0.8828 | 0.7748 |
| empty | 0.9987 | 0.9988 | — | — |
| game2 (per-sq / piece) | 0.9935 / 0.9834 | — | — | — |
| game6 (per-sq / piece) | 0.9614 / 0.9023 | — | — | — |

- **KEY Q1 — combined > sequential within DINO: YES** (stage5 0.9761/0.9377 > stage3 0.9588/0.8908;
  piece-only +0.047). Same pattern ResNet and ConvNeXt both showed → **architecture-independent**.
- **KEY Q2 — DINO stage5 > ConvNeXt stage5: YES** (0.9761/0.9377 vs 0.9557/0.8828; piece-only +0.055).
- **DINOv2 is the best backbone in every regime** (zero-shot piece-only, stage3 both axes, stage5
  both axes). game6 (the shared warp ceiling) also improved markedly: piece-only 0.824 (stage3) →
  **0.902** (stage5).

## Rare-class recovery — combined FIXES the stage3 bB/bQ regression (prediction confirmed)

stage3 had regressed on black bishop/queen (thin in the real set). Reintroducing the full synth
set (rich in rare pieces) via the 50/50 sampler recovers them strongly, and lifts the other weak
classes too:

| class | zeroshot | stage3 | stage5 | stage3→stage5 |
|---|---|---|---|---|
| **bB** | 0.617 | 0.457 | **0.857** | **+0.400** |
| **bQ** | 0.519 | 0.366 | **0.809** | **+0.443** |
| **wB** | 0.259 | 0.443 | **0.701** | **+0.259** (the class weak everywhere — combined finally helps) |
| wQ | 0.205 | 0.644 | 0.864 | +0.220 |
| wN | 0.006 | 0.781 | 0.899 | +0.118 |

→ The stage3 rare-class regression was a **data-sparsity artifact of the thin real set, fixed by
combined training** — not a model limitation. Minor trades the other way (kings dip slightly:
wK 0.899→0.775, bK 0.846→0.799; bR 0.972→0.915) but net piece-only is far higher. King/knight
recovery from zero-shot holds throughout.

## Forgetting Δ (framing)

Raw Δ = **+0.3526** (synth_monitor 0.6473 → 0.9998). This is **acquisition**, not forgetting
(source = DINOv2-pretrained, chess-naive) — do NOT compare to stage3's true-forgetting −0.0214 or
to the ResNet/ConvNeXt true-forgetting numbers. The meaningful number is **synth_monitor_after =
0.9998** (vs dino_zeroshot ~0.9999): combined training retains synth essentially fully, as expected
(ConvNeXt stage5 retained ~0.999 likewise).

## Artifacts

`checkpoints/dino_combined/{best_real.pt(ep19), best_synth_monitor.pt, latest.pt}`;
`results/dino_combined/{training_log.csv, recipe.json, games_2_6_eval.json, game7_results.json,
synth_monitor_results.json, predictions/*.npy}`; `plots/dino_combined/*.png`.

## Next

`dino_combined_linprob` — backbone FROZEN all epochs, head only, on combined synth+real 50/50 (same data
as stage5), select game7. The DINO-only frozen-backbone row + the stage5-vs-linprobe backbone-FT
ablation (how much of DINO's win comes from FT vs frozen features). A summary.md will be written
for it as well.
