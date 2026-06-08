# Checkpoints

Trained model weights for the project. To keep the repository lightweight, the
`.pt` weight files are **not committed to git** — they are hosted externally and
downloaded on demand:

> **Weights (all runs):** _<add Google Drive link here>_

The one exception is the graded checkpoint, which is bundled directly with the
evaluation deliverable at [`../submission/checkpoints/best_real.pt`](../submission/checkpoints)
so `predict_board` runs offline with no download.

The runs are catalogued below; download a run's `.pt` from the link above to
reproduce its evaluation. Locally, weights are organised one folder per run —
`checkpoints/<run_name>/{best_real.pt, latest.pt, best_synth_monitor.pt, ...}`
(the dino training/eval scripts read and write here) — but the `.pt` files
themselves are gitignored. The architecture is **DINOv2 ViT-S/14 + a linear
13-class head** (~22M params) unless noted. Per-run training code lives under
[`../training/dino/`](../training/dino).

## DINOv2 runs

Checkpoint files per run: `best_real.pt` (selected on the real validation game),
`latest.pt` (final epoch), `best_synth_monitor.pt` (best on the synthetic monitor
slice). The zero-shot run has `best_synth.pt` instead of `best_real.pt`.

| Run | Regime | Key checkpoint | Held-out test | Per-square | Piece-only |
|-----|--------|----------------|---------------|-----------:|-----------:|
| `dino_zeroshot` | Synthetic-only (no real data) | `best_synth.pt` | games 2,6 | 0.7800 | 0.5479 |
| `dino_combined_linprob` | Frozen backbone, linear probe on combined data | `best_real.pt` | games 2,6 | 0.7556 | 0.4197 |
| `dino_fine_tuned` | Sequential fine-tune (zero-shot → real) | `best_real.pt` | games 2,6 | 0.9588 | 0.8908 |
| `dino_fine_tuned_Game2boosted` | Stage-3 variant split (val = game6) | `best_real.pt` | game 7 | 0.9750 | 0.9485 |
| `dino_combined` | Combined synth+real (standard split) | `best_real.pt` | games 2,6 | 0.9761 | 0.9377 |
| `dino_combined_Game2boosted` | Combined, variant split (train 4/5/2, val game6) | `best_real.pt` | game 7 | 0.9849 | 0.9689 |
| **`dino_combined_Game6boosted`** | **Combined, train 4/5/6, val game2 — SHIPPED** | **`best_real.pt`** | **game 7** | **0.9858** | **0.9708** |

`dino_combined_Game6boosted/best_real.pt` (epoch 16) is the checkpoint used by the graded
`predict_board`; it is the strongest generaliser on the only game held entirely out
of every combined run (game 7).

### Ablations
| Run | What it isolates |
|-----|------------------|
| `dino_realOnly` | Training on real frames only (no synthetic) — see [`../training/dino/dino_realOnly`](../training/dino/dino_realOnly) |
| `dino_combined_Game6boosted_ablation_LabelSmoothing` | Label-smoothing on the combined recipe — see [`../training/dino/dino_combined_Game6boosted_ablation_LabelSmoothing`](../training/dino/dino_combined_Game6boosted_ablation_LabelSmoothing) |

## Other architectures (baselines / comparison)

ConvNeXt-Tiny and ResNet-18 variants were trained for comparison; their weights and
results live under [`../training/convnext`](../training/convnext) and
[`../training/resnet18`](../training/resnet18). (ConvNeXt checkpoints are ~106 MB each
and, like the rest, are hosted on Drive rather than committed.)
