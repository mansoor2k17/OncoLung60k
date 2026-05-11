# Reproducing Paper Results

Step-by-step guide to reproduce every number, table, and figure in the paper.

## Prerequisites

1. **Clone and install** (see [INSTALL.md](INSTALL.md))
2. **Download dataset** to `data/oncolung60k/` (see [DATASET.md](DATASET.md))
3. **Verify GPU access** with `python scripts/smoke_test.py`

## Quick Reproduction (One Command)

```bash
bash scripts/reproduce_paper.sh
```

Estimated runtime: **5–7 days** on a single NVIDIA RTX 6000 (24 GB).

## Step-by-Step Reproduction

### Step 1: Verify Splits

```bash
python scripts/verify_no_leakage.py --csv splits/oncolung60k_5fold.csv
```

Expected: `OK: No patient leakage across 5 folds`.

### Step 2: Train Modified ConvNeXt (Headline Result)

```bash
python -m src.kfold_cv \
    --config configs/oncolung_modified_convnext.yaml \
    --data_dir data/oncolung60k \
    --output_dir runs/cv_modified_convnext
```

Estimated runtime: ~20 hours on RTX 6000.

Expected output (saved to `runs/cv_modified_convnext/cv_summary.csv`):

| Metric | Mean | Std |
|--------|------|-----|
| accuracy | 0.913 | 0.009 |
| f1_macro | 0.910 | 0.010 |
| auc_macro_ovr | 0.974 | 0.006 |

### Step 3: Train SOTA Baselines

```bash
python -m src.benchmark \
    --configs_dir configs/ \
    --data_dir data/oncolung60k \
    --output_dir runs/sota_benchmark
```

Estimated runtime: ~5 days (8 baselines × 5 folds × ~10 hours each).

Expected results match paper Table 5:

| Model | Accuracy |
|-------|----------|
| ResNet50 | 0.810 ± 0.015 |
| DenseNet121 | 0.851 ± 0.011 |
| EfficientNet-B0 | 0.823 ± 0.013 |
| ViT-B/16 | 0.861 ± 0.014 |
| Swin-B | 0.881 ± 0.011 |
| ConvNeXt-Tiny | 0.832 ± 0.011 |
| ConvNeXt-Small | 0.838 ± 0.010 |
| ConvNeXt-Base | 0.847 ± 0.010 |
| **Modified ConvNeXt** | **0.913 ± 0.009** |

### Step 4: Run Extended Ablation

```bash
for config in configs/ablation/*.yaml; do
    name=$(basename "$config" .yaml)
    python -m src.kfold_cv \
        --config "$config" \
        --data_dir data/oncolung60k \
        --output_dir "runs/ablation/$name"
done
```

Estimated runtime: ~3 days (8 configs × 5 folds).

Expected results match paper Table 7:

| Configuration | Accuracy |
|--------------|----------|
| Base ConvNeXt | 0.812 ± 0.010 |
| + MSP | 0.843 ± 0.010 |
| + CE | 0.844 ± 0.011 |
| + FF | 0.853 ± 0.010 |
| + MSP + CE | 0.867 ± 0.010 |
| + MSP + FF | 0.875 ± 0.010 |
| + CE + FF | 0.879 ± 0.010 |
| **Full ECB** | **0.913 ± 0.009** |

### Step 5: Statistical Significance Tests

```bash
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd
from src.utils.metrics import paired_significance_tests, bonferroni_alpha

# Load per-fold results from each baseline
runs = {}
for d in Path('runs/sota_benchmark').iterdir():
    f = d / 'per_fold_metrics.csv'
    if f.exists():
        runs[d.name] = pd.read_csv(f)['accuracy'].values

# Pairwise tests
ours = runs['oncolung_modified_convnext']
results = []
for name, scores in runs.items():
    if name == 'oncolung_modified_convnext': continue
    res = paired_significance_tests(ours, scores, 'Ours', name)
    results.append(res)

import json
print(json.dumps(results, indent=2, default=str))
print(f'\\nBonferroni-adjusted alpha: {bonferroni_alpha(36, 0.05):.4f}')
"
```

### Step 6: LungHist700 Cross-Dataset

```bash
python -m src.kfold_cv \
    --config configs/lunghist700_modified_convnext.yaml \
    --data_dir data/lunghist700 \
    --output_dir runs/lunghist700
```

Expected: 0.968 ± 0.011 accuracy.

### Step 7: Grad-CAM Visualisations

```bash
mkdir -p data/sample_test_images
# Copy 50 representative test images here

python -m src.utils.explainability \
    --weights runs/cv_modified_convnext/fold0/best.pt \
    --images data/sample_test_images/ \
    --output_dir runs/gradcam_outputs/
```

## Tolerances for Reproduction

Due to CUDA non-determinism and minor differences in environment, reproduction tolerances are:

| Metric | Expected tolerance |
|--------|--------------------|
| Accuracy | ±0.5% |
| Macro F1 | ±0.5% |
| ROC-AUC | ±0.005 |

If your reproduction falls outside these tolerances, please [open an issue](https://github.com/mansoor2k17/OncoLung60k/issues) with:

- Your hardware and CUDA version
- The full output log
- Your `requirements.txt` resolved versions (`pip freeze`)

## Hardware Recommendations

| Setup | Recommended for |
|-------|-----------------|
| RTX 3090/4090 (24 GB) | Full reproduction |
| RTX A5000 (24 GB) | Full reproduction |
| RTX A6000 (48 GB) | Faster training (larger batches) |
| RTX 3070 (8 GB) | Reduce batch_size to 8 |
| Tesla V100 (16 GB) | Reduce batch_size to 16 |

## Troubleshooting

### "Out of memory" errors

Reduce batch size in your config:
```yaml
batch_size: 16  # default 32; try 8 if still OOM
```

### Different results from paper

Check:
1. Are you using the provided `splits/oncolung60k_5fold.csv` (not regenerating)?
2. Are you using the seed values from the config (default 42)?
3. Is your CUDA version compatible (11.2+)?
4. Are you running k-fold CV (not single train/test split)?

### Slow training

Enable mixed-precision (default):
```yaml
amp: true
```

Or use a more powerful GPU.
