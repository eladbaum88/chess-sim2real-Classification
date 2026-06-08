# Run — `dino_combined_Game2boosted` (DINOv2 ViT-S/14, combined, NEW SPLIT) + mini-experiment verdict

> ⚠️ **NOT comparable to the original dino_fine_tuned/stage5 numbers.** Different test set (**game7
> alone**, ~3,520 sq) and split (game2 in training, game6 as val). Do NOT place in the games-2/6
> table column (0.8908 / 0.9377). The valid comparison is **stage5_5 vs stage3_5 on game7**.

## New split (only change vs original dino_combined)

- **Train:** dataset_v1 synth + real[30 manual (8-11) + game4 + game5 + **game2**]; WeightedRandomSampler
  50/50, 100k draws/epoch. (synth 392,448 sq + real 25,600 sq)
- **Val / selection:** **game6** (5,888 sq) · **Held-out TEST:** **game7** (3,520 sq, clean lineage)
- Flags on `--mode stage5`: `--train_pgn_games 4,5,2 --val_game game6 --test_games 7`. Everything
  else identical to dino_combined (DINOv2 ViT-S/14, 224/256 tokens, AdamW wd0.05 cosine, batch64 seed42,
  head 1e-4 / backbone 1e-5, two-phase phaseA2, patience6, 5% dataset_v1 forgetting probe).

## Setup / source

Source = DINOv2-pretrained (fresh). Pre-FT smoke: synth_monitor before **0.6473** (<0.85 → fresh,
not a chess checkpoint — correct), game6 before 0.0905. RTX 3090; **143.2 min**, ran full 20 epochs.
Leakage gate verified: game7 ∉ train/val, game6 ∉ train, all pairwise-disjoint.

## Trajectory (selection = game6, hard/noisy)

Phase-B onset (ep3) train 0.845→0.974, game6 0.699→0.905 — clean adaptation at 1e-5, no
collapse. game6 wobbled 0.90–0.95 then peaked **0.9509 @ epoch 16** (`best_real.pt`). synth_monitor
held ~0.999 throughout (combined trains on synth). Ran all 20 (small late gains).

## HELD-OUT game7 — the verdict (stage5_5 vs stage3_5, identical test + lineage)

| game7 (held-out) | stage3_5 (sequential) | **stage5_5 (combined)** | gap |
|---|---|---|---|
| per-square | 0.9750 | **0.9849** | +0.0099 |
| **piece-only** | 0.9485 | **0.9689** | **+0.0203** |
| empty | 0.9964 | 0.9979 | +0.0015 |

**WINNER: `dino_combined_Game2boosted` (combined) — by +0.020 piece-only on game7.** Combined beats sequential
on the held-out game, consistent with every prior DINO comparison (the combined>sequential pattern
holds on this fresh split too).

### Bishop/rare-class watch — prediction confirmed (the mechanism behind the win)

The synth half is rich in bishops/queens; it lifts exactly stage3_5's weakest game7 classes:

| class | stage3_5 | stage5_5 | Δ |
|---|---|---|---|
| **wB** | 0.755 | **0.964** | **+0.209** |
| **bB** | 0.737 | **0.879** | **+0.141** |
| **bQ** | 0.836 | 0.855 | +0.019 |

wB and bB jump dramatically — the combined synth exposure is what closes stage3_5's bishop gap, the
mechanism that drives the piece-only win. Minor offsets elsewhere (bK 0.982→0.927, bR 0.964→0.945)
but net piece-only clearly higher. wR/wN stay 1.00, wQ 0.982, empty 0.998.

## Forgetting Δ

+0.3522 (acquisition: synth_monitor 0.6473 → **0.9994**; source = DINOv2-pretrained, NOT true
forgetting — do not compare to stage3_5's −0.0289). Retention = synth_monitor_after 0.9994 (combined
retains synth fully).

## MINI-EXPERIMENT DELIVERABLE

On the new split (game2 added to training, game6 as val, **game7 held-out**), the better of
{sequential FT, combined} for DINOv2 ViT-S/14 is **combined (`dino_combined_Game2boosted`)**, by **+0.020
piece-only** (0.9689 vs 0.9485) / +0.010 per-square (0.9849 vs 0.9750) on game7 — driven by the
synth half recovering the bishop classes (wB +0.21, bB +0.14). These game7 numbers are a separate
mini-experiment and are **not** comparable to the original games-2/6 table.

## Artifacts

`checkpoints/dino_combined_Game2boosted/{best_real.pt(ep16), best_synth_monitor.pt, latest.pt}`;
`results/dino_combined_Game2boosted/{training_log.csv, recipe.json (split metadata), game6_results.json,
heldout_game7_eval.json, synth_monitor_results.json, predictions/*.npy}`; `plots/dino_combined_Game2boosted/*.png`.
