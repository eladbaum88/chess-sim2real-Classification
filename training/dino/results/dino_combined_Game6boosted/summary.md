# Run — `dino_combined_Game6boosted` (DINOv2 ViT-S/14, hard-game-in-training ablation) + session record

> **One-line verdict:** swapping the *hard* game6 into training in place of the *easy* game2 did
> **not** produce a meaningfully more robust model on held-out game7 — it lands within noise of
> `dino_combined_Game2boosted` (+0.0009 per-square / +0.0019 piece-only on the fairly-selected checkpoint). A
> clean **negative/null result**: game7 is already near ceiling and the residual errors are the
> intrinsic blur/look-alike errors that no training-data reshuffle can fix.

---

## Session context — why this run exists (three candidate fixes, all rejected)

This run is the end of a session that investigated the tall-piece (bishop/king/queen) errors of the
headline model `dino_combined_Game2boosted`. Three candidate fixes were considered for those errors; the first
two were rejected **by evidence** before spending a retrain, the third (this run) was the only one
with a real mechanism.

1. **Extended/taller crop (Wölflein trick).** Hypothesis: a centered 100 px crop clips the
   mitre/crown of foreshortened far-rank pieces, so a headless bishop reads as a rook. **Rejected.**
   A wide ±1.5-square crop with the square outlined (see `report/game6 crop check/LEAN_direction.png`)
   showed the pieces **fully framed inside the square** in almost every error — the tops are not
   clipped. The confusion is intrinsic look-alike similarity worsened by motion blur, not a cropping
   artifact. An extended crop would shrink the target and add neighbour-piece distractors.

2. **Contrastive / triplet (metric) loss** to push K/Q/B apart. **Rejected by a confidence
   diagnostic.** On game6's 250 tall-piece errors the model is *confidently wrong*: mean confidence
   in the wrong class **0.777** (median 0.816), probability assigned to the true class median
   **0.074**, and only **9 %** of errors are near the decision boundary (conf < 0.5). Metric losses
   widen margins — they help boundary cases, of which there are almost none here. And the *same
   weights* separate these classes at **0.98** on the sharp game7 (vs failing on game6), so the
   embedding space is **not collapsed**. The bottleneck is **destroyed information (blur)**, not
   separability → no metric-learning ablation warranted (consistent with the standing `dino_fine_tuned` /
   `dino_combined_linprob` conclusion).

3. **Class-imbalance fix (weighted CE / balanced sampler).** Judged **low value** for the same two
   reasons: game7 shows the rare classes are already well learned (wK/wQ 0.98), and the errors are
   confident, not borderline — reweighting only nudges borderline cases.

→ The only intervention that adds **new information** rather than reshuffling what the model already
has is **showing it the hard, blurry game6 crops with correct labels in training.** That is this run.

---

## Setup (identical to `dino_combined_Game2boosted` except the game2↔game6 swap)

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14 (`hub:dinov2_vits14`), 22.06 M params |
| Source | DINOv2-pretrained + fresh head (combined/stage5 regime) |
| Recipe | AdamW, wd 0.05, cosine (T_max=18, eta_min=0.01·lr_head), batch 64, seed 42; head lr 1e-4, backbone lr 1e-5; Phase A ep1–2 head-only → Phase B ep3+ all unfrozen |
| Input | 100×100 crops → resize 224 (256 tokens) → ImageNet-normalize |
| Aug | jitter@0.7 → shear@0.8 (±8°) → noise@0.5 — **current aug only** (no blur added) |
| Data | combined: synth `dataset_v1` (392,448 sq) + real (**26,560 sq**), WeightedRandomSampler 50/50, 100k draws/epoch |
| **Epochs / early stop** | **20 epochs, early-stop DISABLED (`patience=0`)** — ran the full 20 so an easy/saturated val game could not stop us early |
| Script | `dino/training_scripts/train_combindedGame6_diag.py` (copy of `train.py` + per-epoch game7 diagnostic; canonical `train.py` untouched) |
| Hardware | RTX 3090, ~6.6 min/epoch |

### The split (the single experimental variable)

| role | games | note |
|---|---|---|
| **Train (real)** | 30 manual (8–11) + game4 (184) + game5 (109) + **game6 (92)** | game6 swapped IN (was val in stage5_5) |
| **Val / selection** | **game2** (77 frames / 4,928 sq) | game2 swapped to val (was a train PGN game in stage5_5) |
| **Held-out test** | **game7** (55 frames / 3,520 sq) | unchanged; clean (∉ train, ∉ val) |

vs `dino_combined_Game2boosted`: train 4,5,**2** / val **game6** / test game7. So both the training set *and* the
val game flip — it is a hard↔easy swap of which real game is trained-on vs selected-on.

### game7 used for diagnosis ONLY

game7 was evaluated **every epoch** for post-hoc curve analysis and a separate best-by-game7
checkpoint. It **never** fed selection, gradients, or early-stopping: `select_acc = game2_real_val`,
the optimizer never saw game7, and the (disabled) early-stop keyed only off game2. Verified in code
and in the log (`select(game2_real_val)=…` every epoch).

---

## Training trajectory (game2 selection vs game7 diagnostic)

| epoch | phase | game2 val (SELECTION) | game7 per-sq (diag) | game7 piece (diag) |
|---|---|---|---|---|
| 1 | A | 0.7851 | 0.6574 | 0.4174 |
| 2 | A | 0.7961 | 0.6960 | 0.5006 |
| 3 | B | 0.9860 | 0.9659 | 0.9263 |
| 4 | B | 0.9915 | 0.9773 | 0.9517 |
| 6 | B | 0.9864 | 0.9801 | 0.9606 |
| 8 | B | 0.9923 | 0.9858 | 0.9708 |
| 13 | B | 0.9905 | 0.9844 | 0.9676 |
| 14 | B | 0.9911 | 0.9855 | 0.9695 |
| **16** | B | **0.9935** (game2 peak → selected) | 0.9858 | 0.9708 |
| 18 | B | 0.9923 | 0.9872 | 0.9740 |
| **19** | B | 0.9927 | **0.9898** (game7 peak) | **0.9797** |
| 20 | B | 0.9933 | 0.9881 | 0.9765 |

- **Phase-B onset (ep3)** jumped game7 0.696 → 0.966 — strong clean adaptation at backbone lr 1e-5,
  same as every prior DINO FT run.
- **Converged by ~ep16.** game2 saturated to ~0.993 by ep4 and only oscillated thereafter; game7
  oscillated in a 0.983–0.990 band with a mild late peak at ep19, then dipped at ep20. **No upward
  trend in the last 5 epochs → more epochs would not help** (the LR is at its cosine floor; the
  residual error is the intrinsic blur error, not a training-time deficit).

---

## Held-out game7 — comparison to `dino_combined_Game2boosted`

| model (held-out game7) | selected by | epoch | per-square | piece-only |
|---|---|---|---|---|
| `dino_combined_Game2boosted` (baseline) | game6 | — | 0.9849 | 0.9689 |
| **`dino_combined_Game6boosted` — game2-selected (fair)** | game2 | 16 | **0.9858** | **0.9708** |
| `dino_combined_Game6boosted` — best-ever game7 (circular) | game7 | 19 | 0.9898 | 0.9797 |

- **Fair comparison** (each model selected on its own held-out val game): **+0.0009 per-square /
  +0.0019 piece-only** over stage5_5. On 3,520 squares, +0.0009 ≈ **3 squares — noise.**
- The best-ever-game7 checkpoint (ep19) is +0.0049 / +0.0108, but it is selected *on the test set*
  → circular, not a generalization claim. Not counted.
- Other health: game2 val (now easy val) final **0.9935 / 0.9834** (confirms game2 is the easy game);
  forgetting Δ = **+0.3526** (synth_monitor 0.6473 → 0.9998 — acquisition, synth retained, healthy,
  same as stage5_5).

### Per-class on game7 — the swap *redistributes* accuracy, it doesn't lift it

| class | stage5_5 | combindedGame6 | Δ |
|---|---|---|---|
| **bB** | 0.879 | **0.929** | **+0.050** |
| **bQ** | 0.855 | **0.927** | **+0.073** |
| bK | 0.927 | 0.927 | 0.000 |
| wQ | 0.982 | 0.982 | 0.000 |
| wK | 0.982 | 0.982 | 0.000 |
| **wB** | 0.964 | **0.909** | **−0.055** |

Putting game6 in training helped exactly its rich-in-hard-cases classes — **black bishop/queen
recover** — but the offsetting loss of game2 from training **cost white bishop**. The hard-piece
accuracy is *moved around*, not increased; net per-square is flat.

---

## Selection-confound check (the user's explicit concern)

Was the easy game2 a bad selection signal that stopped us at a worse checkpoint? **Directionally yes,
in size small.**

- game2 saturated by ep4 (~0.986–0.994, no trend) → a poor discriminator; its "best" at ep16 was
  nearly arbitrary among saturated values.
- The game2-selected checkpoint (ep16) scored game7 **0.9858 / 0.9708**; the true game7 peak (ep19)
  was **0.9898 / 0.9797**.
- **Selection gap = +0.0040 per-square / +0.0089 piece-only.** Below the 1 % per-square flag
  threshold (`selection_confound_game7.json` reports `[ok] no material confound`), but the piece-only
  gap sits right at the edge. So game2-selection *did* leave ~0.4–0.9 % on the table — the mechanism
  the user flagged is real, but the practical penalty is small because game7 is near ceiling anyway.

---

## Takeaways

1. **Hard-game-in-training is a null result for game7.** Training on game6 instead of game2 does not
   make the model more robust on unseen-style game7 — within noise of stage5_5. game7 is already at
   ~0.98–0.99; there is no headroom for a training-data reshuffle to demonstrate robustness.
2. **The residual error is information-limited, not training-limited.** Consistent with the confidence
   diagnostic: the last ~1–2 % are confident, blurry-piece misclassifications. No crop change, metric
   loss, class weighting, additional epochs, or hard-game swap recovers information absent from the
   pixels. The lever for beating this ceiling is input quality (deblur / higher-res capture) or more
   *diverse* real data — not more passes or rearrangement of the same near-ceiling signal.
3. **`dino_combined_Game2boosted` remains the headline DINO model** (combined, game7 0.9849 / 0.9689); this run
   adds a defensible negative control, not a new best.
4. **Easy validation games are weak selectors.** game2 saturated and mildly under-selected (ep16 vs
   the ep19 game7 peak). For future runs prefer a harder/aggregated real val signal.

---

## Artifacts

- `dino/checkpoints/dino_combined_Game6boosted/{best_real.pt (ep16, game2-selected), best_game7_diag.pt (ep19, diagnostic), best_synth_monitor.pt, latest.pt}`
- `dino/results/dino_combined_Game6boosted/{training_log.csv (game2 + per-epoch game7 columns), recipe.json, heldout_game7_eval.json, game2_results.json, selection_confound_game7.json, synth_monitor_results.json, predictions/*.npy}`
- `dino/plots/dino_combined_Game6boosted/{training_curves.png (game2-selection vs game7-diagnostic overlay, with best-epoch markers), game7_cm.png, game2_cm.png, heldout_game7_cm.png}`
- Session crop/confidence diagnostics: `report/game6 crop check/{ERRORS_true_vs_pred.png, FARRANK_rows0-2.png, LEAN_direction.png, RANDOM_variety.png, *_crops.png}`
- Trainer (this run): `dino/training_scripts/train_combindedGame6_diag.py` (canonical `train.py` left untouched)
