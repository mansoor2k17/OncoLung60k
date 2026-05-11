# Hyperparameters Reference

Complete hyperparameter specification for all experiments in the paper.

## Default Training Configuration

These values are used for **all** experiments unless explicitly overridden in a config.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Optimizer | AdamW | Standard for ConvNeXt-family architectures |
| β₁ (Adam) | 0.9 | Default |
| β₂ (Adam) | 0.999 | Default |
| Initial LR | 1×10⁻⁴ | Standard for ImageNet-pretrained fine-tuning |
| Weight decay | 1×10⁻³ | Higher than ImageNet default to compensate for smaller dataset |
| LR schedule | Cosine annealing | Smooth decay |
| Min LR | 1×10⁻⁶ | End of cosine schedule |
| Batch size | 32 | Maximum that fits on 24 GB VRAM |
| Epochs | 100 | Sufficient for convergence with early stopping |
| Early stopping patience | 10 | On validation loss |
| Loss | Cross-entropy | Standard multi-class |
| Label smoothing | 0.1 | Prevents overconfidence |
| Gradient clipping | ‖g‖₂ ≤ 1.0 | Stability with high-contrast patches |
| Mixed precision | fp16 (AMP) | Memory efficiency |

## Architecture Hyperparameters

### Modified ConvNeXt (Base configuration)

| Parameter | Value |
|-----------|-------|
| Stem | 4×4 conv, stride 4 |
| Stage 1 | 3 ECB blocks, dim=128 |
| Stage 2 | 3 ECB blocks, dim=256 |
| Stage 3 | 9 ECB blocks, dim=512 |
| Stage 4 | 3 ECB blocks, dim=1024 |
| Head | Global avg pool + Linear(1024 → 4) |
| Total parameters | 92.4 M |
| FLOPs (256×256 input) | 16.1 G |
| Drop-path rate | 0.1 (linear decay) |
| Layer scale init | 1×10⁻⁶ |

### ECB Components (toggleable)

| Component | Default | Description |
|-----------|---------|-------------|
| `use_msp` | True | Multi-scale pooling (avg + max) |
| `use_ce` | True | Contrast enhancement (max - avg) |
| `use_ff` | True | Feature fusion (concat + 1×1 conv) |

For ablation studies, set any combination of `use_msp/use_ce/use_ff` to control which components are active.

## Augmentation Pipeline

| Augmentation | Probability/Range |
|--------------|-------------------|
| Random horizontal flip | p = 0.5 |
| Random vertical flip | p = 0.5 |
| Random rotation | ±15° |
| ColorJitter brightness | 0.1 |
| ColorJitter contrast | 0.1 |
| ColorJitter saturation | 0.1 |
| ColorJitter hue | 0.05 |
| Macenko stain norm | 25% of training samples |

## Cross-Validation Protocol

| Setting | Value |
|---------|-------|
| Method | Patient-wise StratifiedGroupKFold |
| Number of folds | 5 |
| Stratification variable | Class label |
| Grouping variable | `patient_id` |
| Random seed | 42 (default; varies per fold by `seed + fold`) |
| Inner validation split | 90/10 of training patients |

## Statistical Testing

| Test | Use |
|------|-----|
| Paired t-test | Per-fold accuracy comparison |
| Wilcoxon signed-rank | Non-parametric backup |
| Bonferroni correction | 36 pairwise comparisons (corrected α = 0.0014) |
| 95% confidence intervals | Mean difference, t-distribution-based |

## Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.8.10 |
| PyTorch | 1.10.0 |
| torchvision | 0.11.0 |
| timm | 0.9.12 |
| CUDA | 11.3 (compatible with 11.2 driver) |
| numpy | 1.23.5 |
| pandas | 1.5.3 |
| scikit-learn | 1.2.2 |
| scipy | 1.10.1 |
| matplotlib | 3.7.1 |

Pinned in `requirements.txt`.

## Hardware

Reported training times based on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 6000 (24 GB VRAM) |
| CPU | Intel Xeon, 16 cores |
| RAM | 64 GB |
| Storage | NVMe SSD |
| OS | Ubuntu 20.04 |

Training time per model on OncoLung60K: ~10–14 hours per fold.

## Per-Model Differences

| Model | Image size | Notes |
|-------|-----------|-------|
| Modified ConvNeXt | 256×256 | Default |
| ConvNeXt-Tiny/Small/Base | 256×256 | Same as ours |
| ResNet50 | 256×256 | Standard |
| DenseNet121 | 256×256 | Standard |
| EfficientNet-B0 | 256×256 | Native is 224 but we keep consistent |
| ViT-B/16 | 224×224 | Native input size for ViT |
| Swin-B | 224×224 | Native input size for Swin |

## Reproducibility Notes

Bit-exact reproduction is **not guaranteed** due to:

- CUDA non-determinism in `cudnn.benchmark` mode
- Different floating-point implementations on different GPU architectures
- Multi-threaded data loading order

To maximise determinism:

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This is **not** the default in our scripts because it slows training by 20–30%.
