# Ablation — `LabelSmoothing`: add label smoothing to `dino_combined_Game6boosted`'s loss

> **One-line verdict:** label smoothing (0.1) is a **net negative** here — it lowers accuracy
> (macro-avg 0.9649 → 0.9545, and the look-alike bishops fall ~6–7 pts) **and worsens aggregate
> calibration** (ECE 0.0096 → 0.0836) by inducing global under-confidence. The instructive reason:
> the baseline is **already near-perfectly calibrated** (ECE < 0.01), so the "confidently wrong"
> problem is a **local, rare-blurry-subset** phenomenon, not global overconfidence — and a global
> confidence-dampening regularizer is the wrong instrument for it.

---

## Why this ablation (the reasoning is the point)

`dino_combined_Game6boosted` uses the bare loss: plain `nn.CrossEntropyLoss()`. The session's confidence
diagnostic showed the model fails by being **confidently wrong** on look-alike pieces (mean ~0.8
confidence on errors, true-class prob ~0.07, only ~9% near-boundary) and **information-destroyed**
(motion blur), **not** class-imbalanced. The loss element that targets overconfidence is **label
smoothing**. The honest hypothesis: it likely won't recover accuracy (information is gone on blurry
pieces) but should improve **calibration** — so judge it on ECE + confidence-on-errors, not just
accuracy. Diagnosis → loss element → right metric.

## Single variable

Training criterion only: `nn.CrossEntropyLoss()` → `nn.CrossEntropyLoss(label_smoothing=0.1)`.
The `evaluate()`/selection criterion stays plain CE. Everything else = combined_game6 recipe.

## Compute + config-identity (single variable PROVEN)

RTX 3090 unavailable this session → 11 GiB GTX 1080 Ti → batch 64 OOMs at Phase B. Used gradient
accumulation **batch 32 × 2 = effective 64**. A hard config-identity gate printed + asserted before
epoch 1 (all PASSED):
- `label_smoothing = 0.1` on training CE; eval crit plain CE;
- effective batch == 64;
- **optimizer-steps-per-epoch this_run = 1563 == combined_game6 (batch 64) = 1563** (proves grad-accum
  32×2 is step-equivalent — the one unverified consequence of the hardware change);
- grad-accum math `(loss/2).backward()`, step every 2 (+final) → gradient magnitude == batch 64;
- combined 50/50 `WeightedRandomSampler` (NOT real-only); train g4/5/6+manual, val game2, test game7,
  20 epochs, patience 0, lr 1e-4/1e-5, seed 42, jitter→shear→noise, select=`game2_real_val`;
- smoke aug-fires `mean|s1-s2| = 0.2959` — **matches combined_game6 exactly** (combined ConcatDataset
  puts the synth crop at index 0), confirming the combined data + augmentation path are byte-identical.

LayerNorm-only model ⇒ grad-accum 32×2 is numerically equivalent to batch 64. So the only intended
difference vs combined_game6 is `label_smoothing`.

## Run

Completed all 20 epochs (333.7 min on the 1080 Ti, `completed_all_epochs`). Selection (game2) peaked
at **epoch 15** (game2 0.9953 / piece 0.9880). Train acc ~0.999; **train loss floored ~0.54** (vs ~0.0
in the no-smoothing runs) — the expected signature of label smoothing (targets aren't one-hot).

## Result — game7, game2-selected, vs combined_game6

| | combined_game6 | +LabelSmoothing | Δ |
|---|---|---|---|
| **macro-average** | 0.9649 | 0.9545 | **−0.0104** |
| piece-only | 0.9708 | 0.9632 | −0.0076 |
| per-square *(empty-dominated, misleading)* | 0.9858 | 0.9824 | −0.0034 |
| **ECE-15bin** *(lower=better)* | **0.0096** | 0.0836 | **+0.0740 (worse)** |
| mean confidence on ERRORS *(lower=better)* | 0.8503 | 0.7695 | −0.0808 |
| mean confidence on CORRECT | 0.9964 | 0.9023 | −0.0941 |
| tall-piece confidence on errors | 0.8078 | 0.7416 | −0.0661 |

### Per-class on game7 — LS hurts the look-alikes it was meant to help

| piece | combined → +LS | Δ |
|---|---|---|
| white bishop (wB) | 0.909 → 0.845 | **−0.064** |
| black bishop (bB) | 0.929 → 0.859 | **−0.071** |
| black queen (bQ) | 0.927 → 0.909 | −0.018 |
| wQ / wK / bK | unchanged | 0.000 |

## Interpretation (the insight)

- **My prediction was wrong, for a diagnosable reason — and that is the finding.** I expected LS to
  improve calibration. It did the one thing it targets (confidence-on-errors dropped 0.85 → 0.77), but
  **net it hurt both axes.**
- **The baseline is already excellently calibrated: ECE = 0.0096.** That reframes the failure: the
  model is *not globally overconfident* — the well-calibrated bulk (55% empty + easy pieces at ~0.996
  conf / ~0.99 acc) dominates ECE. "Confidently wrong" is a **small local subset** (rare, blurry tall
  pieces), too few to move aggregate ECE.
- **Label smoothing is a global tool for a local problem.** It capped confidence everywhere
  (confidence-on-correct crashed 0.996 → 0.902) while accuracy stayed ~0.98 → systematic
  **under-confidence** → ECE *rose* to 0.084 (≈ the under-confidence gap). It damaged the
  well-calibrated majority to nudge a tiny tail.
- **It also shaved accuracy on the look-alikes** (wB −0.064, bB −0.071) — consistent with LS softening
  inter-class margins, which hurts *fine-grained* discrimination between near-identical classes.

## Takeaways

1. **Label smoothing is the wrong loss element for this model** — net negative on accuracy and on
   aggregate calibration. `dino_combined_Game6boosted` (plain CE) remains the better model.
2. **Sharper diagnosis as a byproduct:** the overconfidence is *local* (rare blurry tall subset), not
   global miscalibration (baseline ECE 0.0096). A targeted, not global, intervention would be needed —
   and even then, the underlying information is destroyed by blur.
3. A nuanced, mechanism-backed **negative result** — more informative than a lucky positive.

## Artifacts

- `dino/dino_combined_Game6boosted_ablation_LabelSmoothing/checkpoints/{best_real.pt (ep15, game2-selected), best_synth_monitor.pt, latest.pt}` (no `best_game7_diag.pt` — per-epoch game7 diag was off)
- `dino/dino_combined_Game6boosted_ablation_LabelSmoothing/results/{labelsmooth_vs_combined_compare.json, recipe.json, heldout_game7_eval.json, game2_results.json, synth_monitor_results.json, training_log.csv, predictions/*.npy}`
- `dino/dino_combined_Game6boosted_ablation_LabelSmoothing/plots/{training_curves.png, game7_cm.png, game2_cm.png, heldout_game7_cm.png, synth_monitor_cm.png}`
- Trainer: `dino/training_scripts/train_labelsmooth_ablation.py` (`--label_smoothing 0.1 --grad_accum 2`; canonical `train.py` untouched)
- Calibration builder: `dino/dino_combined_Game6boosted_ablation_LabelSmoothing/make_calib_compare.py`

> **Note:** the run completed all 20 epochs and saved every result, but a leftover diag-confound block
> (expected `best_game7_diag.pt`, absent under `--diag_game7 0`) raised `FileNotFoundError` before the
> trainer's own Cell-16 `recipe.json` write. The trainer has since been fixed (confound block gated on
> `diag_game7`), and `recipe.json` here was reconstructed from the run log + comparison JSON.
