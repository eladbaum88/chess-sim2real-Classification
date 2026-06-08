# Model Results — Master Comparison Table

_Chess square / board-state classifier — sim-to-real generalization (BGU Intro to DL, Project 2)._
_Compiled 2026-06-03 from each run's `summary.md` / `games_2_6_eval.json` / `game7_results.json`._

---

## How to read this

Three regimes × three backbones, plus a real-only reference and ablations.

- **Synth-test** = held-out synthetic FENs (in-domain ceiling).
- **game7** = single real game (55 frames / 3,520 sq). For the **zero-shot** rows it is a clean
  monitor; for the **fine-tune / combined** rows it is the **checkpoint-selection signal** (NOT a
  clean held-out test for those — read it as a training-time monitor).
- **Held-out games 2/6** (169 frames / 10,816 sq) is the **clean, architecture-comparable** test set.
  Some early ResNet rows were only scored on games 2/4/5/6 (because games 4/5 weren't in their
  training); for those the matched 2/6 number comes from the eval-only re-eval bridge where it exists.
- **Per-square** counts all 64 squares (empty-dominated, ~69%); **piece-only** excludes the empty
  class and is the honest difficulty axis.
- **Forgetting Δ** = synth-monitor accuracy after training − before (5% slice of `dataset_v1`).
  Negative = catastrophic forgetting. For combined/zero-shot runs starting from ImageNet/DINOv2 the
  Δ is *acquisition* (positive), not forgetting — flagged inline.

---

## 1. Headline table — all models, in order

| # | Model | Backbone | Regime | Real train | Synth-test | game7 | Held-out test set | Per-sq | Piece-only | Forget Δ |
|--:|-------|----------|--------|-----------|:----------:|:-----:|-------------------|:------:|:----------:|:--------:|
| 1 | **Real-only** | ResNet-18 | real-only | games 2/4/5/6 | — | **0.8926**¹ | game7 (held-out) | 0.8926 | — | n/a |
| 2 | **Zero-shot (v1 baseline)** | ResNet-18 | synth-only | — | 0.9991 | 0.5670 | games 2/6 (matched) | 0.5138 | 0.2272 | n/a (synth-only) |
| 2a | Zero-shot + heavy aug | ResNet-18 | synth-only | — | 0.9993 | 0.5213 | — | — | — | n/a |
| 2b | Zero-shot v1.5 (+shear, +legacy data) | ResNet-18 | synth-only | — | 0.9997 | 0.5213 | games 2/4/5/6 | 0.5643 | 0.1926 | n/a |
| 3 | Stage 1 — FT 10 frames | ResNet-18 | fine-tune | 10 frames | 0.9101 | 0.7634 | games 2/4/5/6 | 0.8295 | 0.5012 | −0.0897 |
| 4 | Stage 2 — FT 30 frames | ResNet-18 | fine-tune | 30 frames | 0.9202 | 0.8037 | games 2/4/5/6 | 0.8582 | 0.5408 | −0.0796 |
| 5 | **Stage 3 — FT 323 frames** | ResNet-18 | fine-tune | 30 + g4 + g5 PGN | 0.8705 | 0.9386 | games 2/6 | **0.9085** | **0.7556** | −0.1293 |
| 6 | Stage 4 — combined 30 | ResNet-18 | combined | synth + 30 frames | 0.9025 | 0.7986 | games 2/4/5/6 | 0.8535 | 0.5124 | +0.6951² |
| 7 | **Stage 5 — combined 323** | ResNet-18 | combined | synth + 30 + g4 + g5 | 0.9921 | 0.9517 | games 2/6 | **0.9160** | **0.7748** | +0.7846² |
| 8 | ConvNeXt zero-shot | ConvNeXt-Tiny | synth-only | — | ~0.999 | 0.7438 | games 2/6 | 0.7960 | 0.4621 | n/a (synth-only) |
| 9 | **ConvNeXt Stage 3** | ConvNeXt-Tiny | fine-tune | 30 + g4 + g5 PGN | — | 0.9739 | games 2/6 | **0.9468** | **0.8589** | −0.0360 |
| 10 | **ConvNeXt Stage 5** | ConvNeXt-Tiny | combined | synth + 30 + g4 + g5 | ~0.999 | 0.9807 | games 2/6 | **0.9557** | **0.8828** | +0.5910² |
| 11 | DINOv2 zero-shot | DINOv2 ViT-S/14 | synth-only | — | 0.9999 | 0.6969 | games 2/6 | 0.7800 | 0.5479 | +0.3526² |
| 12 | **DINOv2 Stage 3** | DINOv2 ViT-S/14 | fine-tune | 30 + g4 + g5 PGN | 0.9785 | 0.9795 | games 2/6 | **0.9588** | **0.8908** | −0.0214 |
| 13 | **DINOv2 Stage 5** 🏆 | DINOv2 ViT-S/14 | combined | synth + 30 + g4 + g5 | 0.9998 | 0.9872 | games 2/6 | **0.9761** | **0.9377** | +0.3526² |
| 14 | DINOv2 lin-probe (frozen) | DINOv2 ViT-S/14 | combined, frozen backbone | synth + 30 + g4 + g5 | 0.8733 | 0.7713 | games 2/6 | 0.7556 | 0.4197 | +0.2261² |
| 15 | DINOv2 Stage 3.5 (new split) | DINOv2 ViT-S/14 | fine-tune | 30 + g4 + g5 + **g2** | 0.9710 | —³ | game7 (new split) | 0.9750³ | 0.9485³ | −0.0289 |
| 16 | DINOv2 Stage 5.5 (new split) | DINOv2 ViT-S/14 | combined | synth + 30 + g4 + g5 + **g2** | 0.9994 | —³ | game7 (new split) | 0.9849³ | 0.9689³ | +0.3522² |

¹ Real-only's only held-out test is **game7** (games 2/4/5/6 are its training data), so its number is
not on the games-2/6 column and is not directly comparable to the others' held-out figures.
² Combined / synth-only / linprobe runs start from ImageNet or raw DINOv2, so "Forget Δ" is **synth
acquisition** (the model *gains* synth ability), not catastrophic forgetting. The retention reading
is the high *after* value, not the sign of Δ.
³ Rows 15–16 use a **different split** (game2 moved into training, game6 = val, **game7 = clean
held-out test**). Their per-sq / piece-only are on game7 alone (~3,520 sq) and are **NOT comparable**
to the games-2/6 column above (rows 2, 5, 7–14). See §5 for the matched stage3.5-vs-stage5.5 verdict.

---

## 2. Apples-to-apples — clean held-out **games 2/6**, by regime × backbone

The single comparable table (same 169-frame / 10,816-square test set, same warp + 100 px crop + per-square metric across all three backbones). **Piece-only** is the honest axis.

| Regime | ResNet-18 (per-sq / piece) | ConvNeXt-Tiny (per-sq / piece) | DINOv2 ViT-S/14 (per-sq / piece) |
|--------|:--------------------------:|:------------------------------:|:--------------------------------:|
| **Zero-shot (synth-only)** | 0.5138 / 0.2272 | 0.7960 / 0.4621 | 0.7800 / **0.5479** |
| **Stage 3 (sequential FT)** | 0.9085 / 0.7556 | 0.9468 / 0.8589 | **0.9588 / 0.8908** |
| **Stage 5 (combined)** | 0.9160 / 0.7748 | 0.9557 / 0.8828 | **0.9761 / 0.9377** 🏆 |
| Lin-probe (frozen DINO) | — | — | 0.7556 / 0.4197 |

**Reading:**
- **DINOv2 wins every regime** on the honest piece-only axis (zero-shot 0.548, FT 0.891, combined 0.938).
- **Combined > sequential FT** for all three backbones (architecture-independent result).
- **Backbone fine-tuning is essential**: frozen DINO + linear head reaches only 0.420 piece-only vs
  0.938 with backbone FT (+0.518 gap) — and is even below synth-adapted DINO zero-shot (0.548).
- Backbone ranking in every regime: **DINOv2 > ConvNeXt-Tiny > ResNet-18.**

---

## 3. Catastrophic forgetting — normalization matters

True forgetting Δ on the 5% `dataset_v1` slice, sequential-FT (Stage 3) runs only (those start from a
synth-trained checkpoint, so the Δ is genuine forgetting):

| Backbone | Normalization | Stage-3 forgetting Δ |
|----------|---------------|:--------------------:|
| ResNet-18 | BatchNorm | −0.1293 |
| ConvNeXt-Tiny | LayerNorm | −0.0360 |
| DINOv2 ViT-S/14 | LayerNorm | −0.0214 |

Both LayerNorm backbones forget far less than the BatchNorm ResNet → BN running-stat drift is the
forgetting mechanism. Combined training (Stage 5) sidesteps forgetting entirely by keeping synth in
the mix (synth retained ≈0.99 for all three).

---

## 4. ResNet-18 Stage-3 ablations (`stage3_improved`) — same data, anti-forgetting levers

All on games 2/6, same 323-frame real set; the lever is what changes. Headline = trade-off between
held-out accuracy and forgetting Δ.

| Variant | game7 | Held-out 2/6 per-sq | Piece-only | Forget Δ | Note |
|---------|:-----:|:-------------------:|:----------:|:--------:|------|
| s00 (seed 0, baseline) | 0.9386 | 0.9085 | 0.7556 | −0.1295 | reference |
| s03 (seed 3) | 0.9389 | 0.9067 | 0.7519 | −0.1312 | seed variance |
| s05 (seed 5) | 0.9409 | 0.9051 | 0.7471 | −0.1400 | seed variance |
| l2sp (L2-SP 5e-4) | 0.9386 | 0.9084 | 0.7554 | −0.1297 | no real effect |
| bn-freeze | 0.9347 | 0.8958 | 0.7227 | **−0.0461** | cuts forgetting, costs accuracy |
| rehearsal 0.25 | **0.9449** | 0.8983 | 0.7272 | **−0.0011** | nearly eliminates forgetting, small acc cost |

Takeaway: **rehearsal** (mixing 25% synth back in) nearly zeroes forgetting (−0.001) at a modest
held-out cost — the clean accuracy-vs-retention knob. Combined training (Stage 5) dominates all of
these on raw accuracy anyway.

---

## 5. DINOv2 new-split mini-experiment (NOT comparable to §1–2)

Separate split (game2 moved into training, game6 = val, **game7 = held-out test**). Numbers are on
game7 alone (~3,520 sq) and must **not** be placed in the games-2/6 columns above.

| Model | game7 per-sq | game7 piece-only | Verdict |
|-------|:------------:|:----------------:|---------|
| `dino_fine_tuned_Game2boosted` (sequential) | 0.9750 | 0.9485 | — |
| `dino_combined_Game2boosted` (combined) 🏆 | **0.9849** | **0.9689** | combined wins by +0.020 piece-only |

Confirms the combined > sequential pattern on a fresh split; the win is driven by the synth half
recovering the bishop classes (wB +0.21, bB +0.14).

---

## 6. Per-square accuracy bar (held-out games 2/6, piece-only)

```
DINOv2  Stage 5    ████████████████████████████████████████████  0.9377  🏆
DINOv2  Stage 3    ██████████████████████████████████████████    0.8908
ConvNeXt Stage 5   ██████████████████████████████████████████    0.8828
ConvNeXt Stage 3   █████████████████████████████████████████     0.8589
ResNet  Stage 5    █████████████████████████████████            0.7748
ResNet  Stage 3    ████████████████████████████████             0.7556
DINOv2  zero-shot  ██████████████████████                       0.5479
ConvNeXt zero-shot ███████████████████                          0.4621
DINOv2  lin-probe  █████████████████                            0.4197
ResNet  zero-shot  █████████                                    0.2272
```

---

## Source files

- ResNet-18: `ResNet18/{Real_Only,zero_shot,zero_shot_augmentations,zero_shot_v1.5,fine_tuning/*,combined/*}/results/summary.md`
- ConvNeXt-Tiny: `convnext/results/{comparison_report.md, convnext_*/games_2_6_eval.json, convnext_*/game7_results.json}`
- DINOv2: `dino/results/dino_*/summary.md`
