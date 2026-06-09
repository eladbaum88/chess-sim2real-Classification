# Checkpoints

Trained model weights, one folder per run. Each run's **`best_real.pt`**


`evaluation/predict_board.py` loads `dino_combined_Game6boosted/best_real.pt` from here,
offline and with no download.\
Every run is a **DINOv2 ViT-S/14 backbone + a linear 13-class
head**\
(`dino_zeroshot` is synthetic-only and ships
`best_synth.pt` instead of `best_real.pt`.)

## DINOv2 runs

Per-run files: `best_real.pt` (selected on the real validation game), `latest.pt` (final
epoch), `best_synth_monitor.pt` (best on the synthetic monitor slice).

| Run | Regime | Key checkpoint | Held-out test | Per-square | Piece-only |
|-----|--------|----------------|---------------|-----------:|-----------:|
| `dino_zeroshot` | Synthetic-only (no real data) | `best_synth.pt` | games 2,6 | 0.7800 | 0.5479 |
| `dino_combined_linprob` | Frozen backbone, linear probe on combined data | `best_real.pt` | games 2,6 | 0.7556 | 0.4197 |
| `dino_fine_tuned` | Sequential fine-tune (zero-shot → real) | `best_real.pt` | games 2,6 | 0.9588 | 0.8908 |
| `dino_fine_tuned_Game2boosted` | Fine-tune, variant split (val = game6) | `best_real.pt` | game 7 | 0.9750 | 0.9485 |
| `dino_combined` | Combined synth+real (standard split) | `best_real.pt` | games 2,6 | 0.9761 | 0.9377 |
| `dino_combined_Game2boosted` | Combined, variant split (train 4/5/2, val game6) | `best_real.pt` | game 7 | 0.9849 | 0.9689 |
| **`dino_combined_Game6boosted`** | **Combined, train 4/5/6, val game2 — SHIPPED** | **`best_real.pt`** | **game 7** | **0.9858** | **0.9708** |

`dino_combined_Game6boosted/best_real.pt` (epoch 16) is the shipped checkpoint used by
`predict_board`: it is the strongest generaliser on the one game held entirely out of every
combined run (game 7).

### Ablations
| Run | What it isolates |
|-----|------------------|
| `dino_realOnly` | Real frames only, no synthetic — see [`../training/dino/train_realonly_ablation.py`](../training/dino/train_realonly_ablation.py) |
| `dino_combined_Game6boosted_ablation_LabelSmoothing` | Label smoothing on the combined recipe — see [`../training/dino/train_labelsmooth_ablation.py`](../training/dino/train_labelsmooth_ablation.py) |

## Other architectures (comparison)

ConvNeXt-Tiny and ResNet-18 variants were trained as comparison backbones; their weights and
results live on the Drive (ConvNeXt checkpoints are ~106 MB each and, like the rest, are
hosted there rather than committed).
