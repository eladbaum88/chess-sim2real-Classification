# DINOv2 ViT-S/14 vs ConvNeXt-Tiny vs ResNet-18 — architecture comparison (games 2/6)

| model (games 2/6) | ResNet per-sq | ConvNeXt per-sq | DINOv2 per-sq | ResNet piece | ConvNeXt piece | DINOv2 piece | DINOv2 forget Δ |
|---|---|---|---|---|---|---|---|
| synth-only (zero-shot) | 0.5138 | 0.7960 | 0.7800 | — | 0.4621 | 0.5479 | 0.3526 |
| real fine-tune (Stage 3) | 0.9085 | 0.9468 | 0.9588 | 0.7556 | 0.8589 | 0.8908 | -0.0214 |
| combined (Stage 5) | 0.9160 | 0.9557 | 0.9761 | 0.7748 | 0.8828 | 0.9377 | 0.3526 |
| linear-probe (frozen DINO) | — | — | 0.7556 | — | — | 0.4197 | 0.2261 |

## Per-class held-out (games 2/6) accuracy — DINOv2

| run | wP | wR | wN | wB | wQ | wK | bP | bR | bN | bB | bQ | bK | empty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synth-only (zero-shot) | 0.597 | 0.719 | 0.006 | 0.259 | 0.205 | 0.071 | 0.784 | 0.519 | 0.520 | 0.617 | 0.519 | 0.000 | 0.917 |
| real fine-tune (Stage 3) | 0.993 | 0.990 | 0.781 | 0.443 | 0.644 | 0.899 | 0.995 | 0.972 | 0.918 | 0.457 | 0.366 | 0.846 | 0.999 |
| combined (Stage 5) | 0.994 | 0.990 | 0.899 | 0.701 | 0.864 | 0.775 | 0.994 | 0.915 | 0.959 | 0.857 | 0.809 | 0.799 | 0.999 |
| linear-probe (frozen DINO) | 0.635 | 0.590 | 0.444 | 0.080 | 0.015 | 0.337 | 0.383 | 0.382 | 0.427 | 0.314 | 0.023 | 0.160 | 0.954 |

## Recipe (each architecture done right)

ResNet-18: SGD + two-phase freeze. ConvNeXt-Tiny (~27.8M): AdamW + cosine + ConvNeXt two-phase freeze. DINOv2 ViT-S/14 (~21M): AdamW + cosine, ViT two-phase freeze, inputs resized 100->224 (256 patch tokens) before ImageNet-normalize, backbone LR 1e-5 (ViT FT is fragile). LayerNorm only -> BN-freeze N/A. linprobe = frozen backbone, head only.

- **synth-only (zero-shot)** (`dino_zeroshot`): AdamW, lr_head=0.0001, lr_backbone=1e-05, input=224x224 (256 tokens), epochs=10, select synth_val @ ep 10, load=hub:dinov2_vits14.

- **real fine-tune (Stage 3)** (`dino_fine_tuned`): AdamW, lr_head=0.0001, lr_backbone=1e-05, input=224x224 (256 tokens), epochs=20, select game7_real_val @ ep 7, load=hub:dinov2_vits14.

- **combined (Stage 5)** (`dino_combined`): AdamW, lr_head=0.0001, lr_backbone=1e-05, input=224x224 (256 tokens), epochs=20, select game7_real_val @ ep 19, load=hub:dinov2_vits14.

- **linear-probe (frozen DINO)** (`dino_combined_linprob`): AdamW, lr_head=0.0001, lr_backbone=frozen (linprobe), input=224x224 (256 tokens), epochs=20, select game7_real_val @ ep 18, load=hub:dinov2_vits14.
