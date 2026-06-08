# Ablation — `realOnly`: remove the synthetic half of `dino_combined_Game6boosted`

> **One-line verdict:** the synthetic half specifically props up the **rare black tail**. Pulling
> all synth data drops game7 **macro-average 0.9649 → 0.9429 (−0.022)**, and the damage is almost
> entirely the **black queen (0.927 → 0.709, −0.218)** plus the black bishop (−0.051); white
> royalty/bishop and black king are unchanged. First **positive** ablation result of this line of
> work — a concrete, defensible justification for the combined recipe.

---

## What this ablates and why

`dino_combined_Game6boosted` (DINOv2 ViT-S/14, combined synth+real; train game4/5/6 PGN + 30 manual,
val game2, test game7) scores game7 per-square 0.9858 / piece-only 0.9708. Its weak classes are the
rare tall look-alikes — wB 0.909, bQ/bK 0.927, bB 0.929 — which are scarce in real data but abundant
in the synthetic crowded boards. This **true ablation** removes the synthetic half and retrains
**real-only**, eval fixed on game7, to measure how much synth carries that tail.

## Single variable — synthetic data removed

- **Training set = real only:** 30 manual frames (games 8–11) + game4/5/6 PGN = **26,560 squares**,
  **zero synthetic samples**.
- **Sampler:** the 50/50 synth/real `WeightedRandomSampler` no longer applies → uniform
  `RandomSampler` over real, `replacement=True, num_samples=100_000`/epoch (epoch length pinned to
  match combined_game6's 100k draws/epoch → same optimizer-step count + cosine schedule).
- Everything else **byte-identical** to combined_game6 (proven by `diff` of the trainer scripts —
  only the flags, sampler branch, output-root redirect, and recipe labels changed): DINOv2 ViT-S/14,
  DINOv2-pretrained + fresh head, 100×100→224 + ImageNet-normalize, val=game2, test=game7, 20 epochs,
  head lr 1e-4 / backbone lr 1e-5, cosine T_max=18, Phase A/B freeze scheme, jitter→shear→noise aug,
  seed 42, `patience=0`. Selection metric = `game2_real_val`; **game7 diagnostic-only** (out of
  selection/gradient/early-stop).

### Compute note — gradient accumulation (numerically equivalent)

The RTX 3090 (24 GiB) used for combined_game6 was reallocated; only an 11 GiB GTX 1080 Ti was
available, which OOM'd at epoch 3 (Phase B, full-backbone backprop at batch 64). Fix: **gradient
accumulation, micro-batch 32 × 2 = effective batch 64.** For a mean-reduction CE loss on a
**LayerNorm-only** model (DINOv2 has no BatchNorm), accumulated grads over 2 micro-batches equal the
single-batch-64 gradient exactly, and the optimizer-step count / cosine schedule are unchanged — so
comparability with combined_game6 is preserved. Phase B then ran at ~6.0/10.9 GiB.

## Isolation audit (logged before training)

`synthetic samples in training = 0`; `real train squares = 26,560`; sampler = uniform RandomSampler
over real, 100k/epoch; `batch=32x2accum(eff=64)`; select_on=`game2_real_val`; seed 42; smoke batch
`(64,3,100,100)` float32 [0,1], labels 0–12. (aug-fires read 0.1203 vs combined's 0.2959 — benign:
the probe reads `train_dataset[0]`, now a real manual crop instead of the synth crop that was first
in the combined ConcatDataset; the augmentation pipeline is byte-identical per the trainer diff.)

## Trajectory (game2 selection vs game7 diagnostic)

| epoch | game2 val (SELECTION) | game7 (diag) |
|---|---|---|
| 1 (A) | 0.7782 | 0.6562 |
| 2 (A) | 0.8034 | 0.7199 |
| 3 (B) | 0.9872 | 0.9722 |
| 7 | 0.9901 | 0.9849 |
| 13 | 0.9923 | 0.9864 |
| **14** | **0.9945** (game2 peak → selected) | 0.9804 |
| 20 | 0.9935 | 0.9852 |

Phase-B onset (ep3) jumped game7 0.72→0.97; both curves then oscillate near ceiling (game2 ~0.99,
game7 ~0.98). Selected checkpoint = ep14 (game2-best).

## Result — game7, game2-selected, vs combined_game6 baseline

| metric | combined_game6 | realOnly | Δ |
|---|---|---|---|
| **macro-average** (mean of 13 per-class) | 0.9649 | **0.9429** | **−0.0221** |
| piece-only | 0.9708 | 0.9587 | −0.0121 |
| per-square *(empty-dominated, misleading)* | 0.9858 | 0.9804 | −0.0054 |
| game2 val (selection) | — | 0.9945 / 0.9857 | — |
| forgetting Δ (synth_monitor) | +0.3526 | +0.1668 (0.6473→0.8141) | — |

### Per-class on game7 — the synth contribution is the rare black tail

| piece | combined → realOnly | Δ | note |
|---|---|---|---|
| **black queen (bQ)** | 0.927 → **0.709** | **−0.218** | ~51/55 → ~39/55 correct (+12 errors on 55 bQ squares) |
| black bishop (bB) | 0.929 → 0.879 | −0.051 | ~5 more errors |
| white bishop (wB) | 0.909 → 0.900 | −0.009 | ~flat |
| wQ / wK / bK | unchanged | 0.000 | real data covers these |

The entire macro-average drop is essentially the **bQ collapse**.

## Interpretation

- **Synth specifically rescues the black queen (and dents on the black bishop).** bQ is ~1.5% of real
  squares; the real training set has too few examples to learn it robustly, so the synthetic crowded
  boards — rich in clean bQ/bB instances — are what hold it up. Remove synth and bQ falls 22 points.
  This is a concrete, defensible reason the combined recipe matters: **synth is not padding; it props
  up the rarest pieces.**
- **The effect is class-selective, not uniform** (honest nuance): wQ/wK/bK are fine without synth, so
  the story is "synth rescues bQ (and bB)," not "synth helps all rare pieces." The white–black queen
  asymmetry (wQ 0.982 unchanged, bQ −0.218) is likely a real-data coverage difference in this split.
- **forgetting Δ = +0.1668** (synth_monitor rose 0.6473→0.8141 *without any synth training*) — this is
  real→synth knowledge transfer: learning real chess pieces partially generalizes to synth renders.
  Lower than combined's +0.3526 (which trains on synth), as expected.
- **Per-square hid everything** (−0.005) — vindicates judging this on macro-average/per-class.

## Takeaways

1. **The synthetic half's measured value = robust rare-black-piece (esp. queen) recognition.** This is
   the first ablation in the line (after crop / contrastive / class-weight / game6-swap, all null) to
   show a clear positive effect.
2. **`dino_combined_Game6boosted` (combined) stays the better model**; real-only is the negative control that
   isolates what synth buys.
3. Reinforces the project's synthetic-to-real thesis with a single-class smoking gun rather than a
   diffuse average.

## Artifacts

- `dino/dino_realOnly/checkpoints/{best_real.pt (game2-selected), best_game7_diag.pt, best_synth_monitor.pt, latest.pt}`
- `dino/dino_realOnly/results/{realonly_vs_combined_compare.json, recipe.json, heldout_game7_eval.json, game2_results.json, selection_confound_game7.json, synth_monitor_results.json, training_log.csv, predictions/*.npy}`
- `dino/dino_realOnly/plots/{training_curves.png, game7_cm.png, game2_cm.png, heldout_game7_cm.png}`
- Trainer: `dino/training_scripts/train_realonly_ablation.py` (`--no_synth --grad_accum 2 --output_root …`; canonical `train.py` untouched)
- Comparison builder: `dino/dino_realOnly/make_compare.py`
