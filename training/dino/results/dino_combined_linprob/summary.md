# Run 4 — `dino_combined_linprob` (DINOv2 ViT-S/14, FROZEN backbone, head-only)

Linear probe: backbone **frozen for all epochs** (no Phase B), train only the 13-class head, on
the SAME combined synth+real 50/50 data as `dino_combined`. Identical to stage5 in every way EXCEPT
the backbone never unfreezes — that single difference is the **backbone-fine-tuning ablation**.
Canonical DINOv2 usage; DINO-only regime (no CNN parallel).

## Setup

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14 (`hub:dinov2_vits14`), FROZEN (raw pretrained, never chess-adapted) |
| Trainable | head only — **5,005 params** (`Linear(384,13)`) |
| Input | 100×100 crops → resize **224** (256 tokens) before ImageNet-normalize |
| Recipe | AdamW, wd 0.05, **cosine over all 20 epochs**, batch 64, seed 42; head lr 1e-4; no Phase B |
| Data | combined synth (392,448) + real (20,672), WeightedRandomSampler 50/50, 100k draws/epoch |
| Selection | game7 real_val, patience 6 |
| Hardware | RTX 3090; **132.5 min**, 20 epochs (full), ~6.5 min/epoch (ViT forward at 224 dominates even frozen) |

## Verification

- Leakage: same combined construction as stage5; combined-train ∩ (game7 ∪ games-2/6) = 0.
- Pre-FT smoke: synth_monitor 0.6473, game7 0.0622 (fresh DINOv2-pretrained; `<0.85` assert fired).
- Phase log: every epoch `phase=A`, 5,005 trainable params — backbone never unfroze. ✓
- ep1–2 matched stage5 exactly (both head-only there); divergence began ep3 where stage5 unfroze.

## Trajectory (selection = game7) — frozen-feature ceiling

game7: 0.661 → 0.701 → 0.711 → 0.724 → 0.742 → 0.752 → 0.760 (ep7) → … plateau ~0.77 →
**peak 0.7713 @ epoch 18** (`best_real.pt`). Slow head-only climb to a hard ceiling; train acc
also capped at ~0.905 (a linear head on frozen features cannot fully fit even the train data).

## THE ABLATION — held-out games 2/6 (linprobe vs stage5, only the freeze differs)

| metric | linprobe (frozen) | stage5 (FT backbone) | gap |
|---|---|---|---|
| per-square | 0.7556 | 0.9761 | **+0.2204** |
| **piece-only** | **0.4197** | **0.9377** | **+0.5180** |
| empty | 0.9536 | 0.9987 | +0.045 |
| game2 (per-sq/piece) | 0.828 / 0.649 | 0.994 / 0.983 | |
| game6 (per-sq/piece) | 0.695 / 0.242 | 0.961 / 0.902 | |

**Result: case (b) — `stage5 ≫ linprobe`, decisively.** Fine-tuning the backbone is doing the
heavy lifting: frozen DINOv2 features + a linear head reach only **0.42 piece-only** vs **0.94**
with backbone FT — a **+0.52 piece-only gap**. Backbone adaptation to the chess domain is essential;
raw self-supervised features are far from sufficient on real boards.

**Sharper still:** linprobe piece-only (0.4197) is **below even `dino_zeroshot` (0.5479)**. The
reason: `dino_zeroshot` fine-tuned the backbone (on synth), so its backbone is chess-adapted;
`dino_combined_linprob` keeps the backbone at raw DINOv2-pretrained (never chess-adapted) and only fits a
head. So *any* backbone adaptation — even synth-only — beats a frozen raw backbone on real chess.
game6 collapses worst (piece-only 0.242): frozen features can't handle the foreshortened far/near
ranks (the shared warp ceiling) without adaptation.

## Per-class (linprobe vs stage5) — frozen features especially fail tall/rare pieces

| | wP | wR | wN | wB | wQ | wK | bP | bR | bN | bB | bQ | bK | empty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linprobe | .635 | .590 | .444 | .080 | .015 | .337 | .383 | .382 | .427 | .314 | .023 | .160 | .954 |
| stage5 | .994 | .990 | .899 | .701 | .864 | .775 | .994 | .915 | .959 | .857 | .809 | .799 | .999 |

Frozen features barely separate queens (wQ 0.015, bQ 0.023), bishops (wB 0.080), kings (bK 0.160) —
the tall-piece cluster confusion that real-data FT resolves in stage5. Confirms the king/bishop/queen
separability requires backbone adaptation, not just a better head — but since stage5 already fixes
it, no contrastive-loss ablation is warranted (consistent with the stage3 finding).

## Forgetting Δ (framing)

Raw Δ = +0.2261 (synth_monitor 0.6473 → 0.8733). Acquisition by the head; the backbone is frozen
so it cannot forget. Note the head-only synth ceiling is just 0.873 (vs stage5's 0.9998) — the
frozen representation limits synth fit too, consistent with the real-side ceiling. Not a forgetting
test.

## Takeaway

DINOv2's strength in this project comes from **fine-tuning the backbone on chess**, not from its
frozen features. The "just use frozen DINO" story does **not** hold here (+0.52 piece-only gap);
the win is backbone adaptation. This makes `dino_combined` (combined, FT) the headline DINO result.

## Artifacts

`checkpoints/dino_combined_linprob/{best_real.pt(ep18), best_synth_monitor.pt, latest.pt}`;
`results/dino_combined_linprob/{training_log.csv, recipe.json, games_2_6_eval.json, game7_results.json,
synth_monitor_results.json, predictions/*.npy}`; `plots/dino_combined_linprob/*.png`.
