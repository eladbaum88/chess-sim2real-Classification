# Run 1 — `dino_zeroshot` (DINOv2 ViT-S/14, synth-only baseline)

Third backbone in the sim-to-real comparison (after ResNet-18 and ConvNeXt-Tiny). Synth-only
zero-shot baseline: trained on the full synthetic `dataset_v1` from DINOv2-pretrained weights,
then evaluated on real boards. This checkpoint is the source `dino_fine_tuned` fine-tunes from.

## Setup

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14, loaded `hub:dinov2_vits14` (cached) |
| Params | 22,061,581 (~22M; between ResNet-18 11.2M and ConvNeXt-Tiny 27.8M) |
| Head | `Linear(384, 13)` on the CLS embedding |
| Input | 100×100 crops resized to **224** (16×16 = 256 patch tokens) immediately before ImageNet-normalize |
| Recipe | AdamW, wd 0.05, cosine (eta_min=0.01·lr_head), batch 64, seed 42 |
| Freeze | Phase A (ep1, head-only) → Phase B (ep2–10, all unfrozen): head lr 1e-4, backbone lr **1e-5** |
| Data | full `dataset_v1` synth, 90/10 by-image split; **selection = synth_val** |
| Norms | LayerNorm only (BN=0) → BN-freeze N/A |
| Hardware | RTX 3090; **258.6 min (~4.3h)**, ~26 min/epoch |

## Leakage / data verification (pre-launch)

- Synth source = `dataset_v1` (NOT v1.5); manifest 392,448 rows / 6,132 unique source images.
- synth-train ∩ (game7 ∪ games-2/6) = **0** — the only train↔eval leak that matters; zero.
- game7 (55 frames) = monitor only; games 2/6 (169 frames) = held-out test only; both eval-only,
  never in the synth training set. Selection signal = synth_val (game7 is NOT used for selection).

## Training trajectory (selection = synth_val)

| epoch | phase | train | synth_val (sel) | synth_monitor | game7 (monitor) |
|---|---|---|---|---|---|
| 1 | A | 0.8586 | 0.8927 | 0.8955 | 0.4412 |
| 2 | B | 0.9898 | 0.9979 | 0.9983 | 0.7364 |
| 3 | B | 0.9966 | 0.9981 | 0.9982 | 0.6727 |
| 4 | B | 0.9978 | 0.9995 | 0.9996 | 0.7943 |
| 5 | B | 0.9985 | 0.9994 | 0.9996 | 0.7290 |
| 6 | B | 0.9991 | 0.9996 | 0.9997 | 0.7733 |
| 7 | B | 0.9994 | 0.9997 | 0.9997 | 0.8011 |
| 8 | B | 0.9997 | 0.9998 | 0.9999 | 0.7085 |
| 9 | B | 0.9998 | 0.9998 | 0.9998 | 0.6861 |
| **10** | **B** | **0.9999** | **0.9999** | 0.9999 | 0.6969 |

- **Selected epoch 10** → `checkpoints/dino_zeroshot/best_synth.pt`.
- **No underfit:** head-only Phase A already linear-probed synth to 0.893; unfreezing the backbone
  at 1e-5 (ep2) jumped synth_val to 0.998 and it saturated. The 1e-5 backbone LR is ample for the
  large synth set — no bump needed (the 1e-5 fragility caution applies to the small real set in stage3).
- Pre-train (DINOv2-init + random head): synth_monitor 0.6473, game7 0.0622 — well below trained
  (the random head leans toward the dominant 'empty' class on DINO features, hence 0.65 not chance-0.08).

## Held-out games 2/6 (the comparable number) — synth-only row

| metric | DINOv2 | ConvNeXt-Tiny | ResNet-18 |
|---|---|---|---|
| per-square | 0.7800 | **0.7960** | 0.5138 |
| **piece-only** | **0.5479** | 0.4621 | — |
| empty | 0.9168 | 0.9928 | — |
| game2 (per-sq / piece) | 0.8330 / 0.6238 | 0.779 / 0.393 | — |
| game6 (per-sq / piece) | 0.7357 / 0.4892 | 0.810 / — | — |

**Reading:** on the honest axis (**piece-only**), DINOv2 zero-shot is the **best of the three**
(0.548 > ConvNeXt 0.462) — self-supervised features transfer to real piece *appearance* better.
Its per-square (0.780) is a hair under ConvNeXt (0.796) only because per-square is empty-dominated
and DINO's empty accuracy is lower (0.917 vs 0.993). Both far exceed ResNet's 0.5138.

Per-class (games 2/6): bP 0.784, wR 0.719, bB 0.617, bN 0.520, empty 0.917 strong; wN 0.006,
bK 0.000, wK 0.071, wQ 0.205 weak — knights/kings/queens hardest zero-shot (same pattern as the CNNs).

## Forgetting Δ

+0.3526 (synth_monitor 0.6473 → 0.9999). This is **acquisition**, not forgetting — the source is
DINOv2-pretrained (chess-naive), so the model *gains* synth ability. True forgetting is measured at
`dino_fine_tuned` (source = this synth checkpoint).

## Artifacts

`checkpoints/dino_zeroshot/{best_synth.pt, best_synth_monitor.pt, latest.pt}`;
`results/dino_zeroshot/{training_log.csv, recipe.json, games_2_6_eval.json, game7_results.json,
synth_monitor_results.json, predictions/*.npy}`; `plots/dino_zeroshot/{training_curves.png, *_cm.png}`.

## Next

`dino_fine_tuned` — sequential fine-tune from `best_synth.pt` (epoch 10) on real data (30 manual +
game4 + game5), select on game7, patience-6, backbone 1e-5. The 1e-5 fragility caution genuinely
applies there (~20k-square real set); watch train-acc trajectory for underfit vs over-adaptation.
