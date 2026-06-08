# Run — `dino_fine_tuned_Game2boosted` (DINOv2 ViT-S/14, sequential FT, NEW SPLIT)

> ⚠️ **NOT comparable to the original dino_fine_tuned/stage5 numbers.** This run uses a DIFFERENT
> test set (**game7 alone**, ~3,520 squares) and a different split (game2 moved into training,
> game6 as val). It must NOT be dropped into the games-2/6 comparison-table column (0.8908 /
> 0.9377). The winner of this mini-experiment is chosen among {stage3_5, stage5_5} on **game7**.

## New split (the only change vs the original dino_fine_tuned)

- **Train real:** 30 manual (games 8-11) + game4 + game5 + **game2** (added) = 400 frames / ~25,600 sq
- **Val / selection:** **game6** (92 frames / 5,888 sq)
- **Held-out TEST:** **game7** (55 frames / 3,520 sq) — clean in this lineage (fresh retrain;
  game7 in neither train nor selection)

Driven by CLI flags on the existing `--mode stage3` (defaults unchanged for the original runs):
`--train_pgn_games 4,5,2 --val_game game6 --test_games 7`. Everything else identical to dino_fine_tuned:
DINOv2 ViT-S/14, 224 resize (256 tokens), AdamW wd0.05 cosine, batch64 seed42, head 1e-4 /
backbone 1e-5, two-phase freeze (phaseA 2), patience 6, 5% dataset_v1 forgetting probe.

## Setup / source

Source = `dino_zeroshot/best_synth.pt` (epoch 10, synth_val 0.9999). Pre-FT smoke: synth_monitor
before **0.9999** (>0.95 → confirms synth-trained weights loaded), game6 before 0.7357. RTX 3090;
**37.2 min**, early-stopped at epoch 12. Leakage gate verified: game7 ∉ train, ∉ val; game6 ∉ train;
train ∩ game6 = train ∩ game7 = game6 ∩ game7 = 0.

## Trajectory (selection = game6, the hard/noisy game)

| epoch | phase | train | game6 (sel) | synth_monitor |
|---|---|---|---|---|
| 1 | A | 0.8506 | 0.8551 | 0.9985 |
| 2 | A | 0.9026 | 0.8682 | 0.9950 |
| 3 | B | 0.9788 | 0.9178 | 0.9848 |
| 5 | B | 0.9944 | 0.9256 | 0.9797 |
| **6** | **B** | 0.9965 | **0.9351** | 0.9710 |
| 7–12 | B | →0.999 | 0.916–0.933 | →0.967 |

- **Selected epoch 6** → `best_real.pt`. Early-stopped at 12 (game6 no improvement over 0.9351 in 6).
- No underfit / no collapse: phase-B onset (ep3) jumped train 0.903→0.979, game6 0.868→0.918 at
  backbone 1e-5 — same clean adaptation as the original DINO FT runs. game6 val curve was wobblier
  than game7 was (expected — game6 is the hard, foreshortened ~92-frame game).

## Held-out game7 (the clean test for this mini-experiment)

| metric | value |
|---|---|
| per-square | **0.9750** |
| **piece-only** | **0.9485** |
| empty | 0.9964 |
| n | 3,520 sq (55 frames, single game) |

Per-class (game7): wR 1.00, wN 1.00, bK 0.982, wQ 0.982, wP 0.99, bP 0.987, bN 0.976, wK 0.964,
bR 0.964; weakest **bB 0.737, wB 0.755, bQ 0.836** (bishops/black-queen still the hard classes).
Single-game test → noisier than the old 2-game number; read with that caveat.

## Forgetting Δ

**−0.0289** (true forgetting: synth_monitor 0.9999 → 0.9710 at ep6) — consistent with the
LayerNorm low-forgetting profile (orig dino_fine_tuned was −0.0214).

## Artifacts

`checkpoints/dino_fine_tuned_Game2boosted/{best_real.pt(ep6), best_synth_monitor.pt, latest.pt}`;
`results/dino_fine_tuned_Game2boosted/{training_log.csv, recipe.json (split metadata), game6_results.json,
heldout_game7_eval.json, synth_monitor_results.json, predictions/*.npy}`; `plots/dino_fine_tuned_Game2boosted/*.png`.

## Next

`dino_combined_Game2boosted` — same new split, combined synth+real (+game2) from DINOv2-pretrained, weighted
50/50, select game6, test game7. The game7 winner is decided between the two.
