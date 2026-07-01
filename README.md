# OncoLung60K & Modified ConvNeXt for Lung Cancer Histopathology

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset DOI](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/14995223)

Official implementation of **"Leveraging the Modified ConvNeXt Model and OncoLung60K Dataset for Lung Cancer Diagnosis"** (under review at *Journal of Advanced Intelligence*).

This repository contains:
- **Modified ConvNeXt** model with Enhanced ConvNeXt Block (ECB) for lung cancer subtype classification.
- **OncoLung60K dataset** specification and patient-wise splits (the dataset itself is hosted on [Zenodo](https://zenodo.org/records/14995223)).
- Full training, evaluation, ablation, and explainability pipelines.
- Patient-wise 5-fold cross-validation protocol with statistical significance testing.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k

# Install dependencies (Python 3.8+, CUDA 11.2+ recommended)
pip install -r requirements.txt

# OR use Docker for full reproducibility
docker build -t oncolung60k -f docker/Dockerfile .
docker run --gpus all -it oncolung60k

# Run a smoke test (5 minutes)
python scripts/smoke_test.py

# Train Modified ConvNeXt on OncoLung60K
python -m src.train --config configs/oncolung_modified_convnext.yaml

# Evaluate with patient-wise 5-fold cross-validation
python -m src.kfold_cv --config configs/oncolung_modified_convnext.yaml
```

---

## Key Features

- **Modified ConvNeXt architecture** with Enhanced ConvNeXt Block (ECB) integrating multi-scale pooling, max–average contrast computation, and hierarchical fusion.
- **Patient-wise 5-fold cross-validation** using `sklearn.model_selection.StratifiedGroupKFold` to eliminate data leakage.
- **Comprehensive evaluation**: ROC-AUC, sensitivity, specificity, PPV, NPV, per-class F1, paired t-test/Wilcoxon significance with Bonferroni correction.
- **Explainability**: Grad-CAM, Grad-CAM++, Score-CAM with quantitative IoU evaluation against pathologist ROIs.
- **8 baseline architectures** for comparison: ResNet50, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny/Small/Base, ViT-B/16, Swin-B.
- **Full reproducibility**: Pinned dependencies, fixed seeds, frozen Docker image, exact patient-wise CSV splits released.

---

## Repository Structure

```
OncoLung60k/
├── README.md                          # This file
├── LICENSE                            # MIT (code) + CC BY-NC 4.0 (dataset)
├── CITATION.cff                       # Machine-readable citation
├── requirements.txt                   # Python dependencies (pinned)
├── pyproject.toml                     # Project metadata
├── .gitignore                         # Python, data, weights ignored
│
├── configs/                           # YAML training configurations
│   ├── oncolung_modified_convnext.yaml
│   ├── oncolung_convnext_base.yaml
│   ├── oncolung_resnet50.yaml
│   ├── oncolung_swin_base.yaml
│   ├── oncolung_vit_base.yaml
│   ├── lunghist700_modified_convnext.yaml
│   └── ablation/                      # 8 ablation configs (MSP/CE/FF combinations)
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── modified_convnext.py       # Main model
│   │   ├── ecb.py                     # Enhanced ConvNeXt Block
│   │   └── builder.py                 # Model builder for baselines
│   ├── data/
│   │   ├── dataset.py                 # PyTorch Dataset
│   │   ├── preprocessing.py           # Tissue masking, patch extraction
│   │   └── augmentation.py            # Stain norm, ColorJitter, etc.
│   ├── utils/
│   │   ├── metrics.py                 # ROC, sens/spec, statistical tests
│   │   ├── explainability.py          # Grad-CAM wrappers
│   │   └── splits.py                  # Patient-wise StratifiedGroupKFold
│   ├── train.py                       # Single training run
│   ├── evaluate.py                    # Single model evaluation
│   ├── kfold_cv.py                    # Full k-fold CV pipeline
│   └── benchmark.py                   # Multi-model SOTA benchmark
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_setup/
│   │   └── 01_dataset_overview.ipynb
│   ├── 02_training/
│   │   └── 02_train_modified_convnext.ipynb
│   ├── 03_evaluation/
│   │   ├── 03a_kfold_cv.ipynb
│   │   ├── 03b_roc_pr_curves.ipynb
│   │   └── 03c_statistical_tests.ipynb
│   └── 04_explainability/
│       ├── 04a_gradcam.ipynb
│       └── 04b_iou_with_roi.ipynb
│
├── splits/                            # Patient-wise CV split CSVs
│   ├── README.md
│   ├── oncolung60k_5fold.csv          # 60K rows: filename, label, patient_id, fold
│   └── lunghist700_5fold.csv          # 691 rows: same schema
│
├── scripts/                           # Helper scripts
│   ├── smoke_test.py                  # 5-min sanity check
│   ├── download_weights.py            # Fetch from Zenodo
│   ├── reproduce_paper.sh             # Full reproduction pipeline
│   └── verify_no_leakage.py           # Sanity check for splits
│
├── docs/                              # Documentation
│   ├── INSTALL.md
│   ├── DATASET.md                     # OncoLung60K specification
│   ├── REPRODUCE.md                   # Step-by-step reproduction guide
│   ├── HYPERPARAMETERS.md             # Full hyperparameter reference
│   └── FAQ.md
│
├── tests/                             # Unit tests
│   ├── test_ecb.py
│   ├── test_splits.py
│   └── test_metrics.py
│
│
└── assets/                            # Images for README/docs
    └── ecb_diagram.png
```

---

## Installation

Pip (recommended for development)

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "from src.models.modified_convnext import ModifiedConvNeXt; print('OK')"
```

### System requirements

- **OS**: Ubuntu 20.04 (tested), macOS, Windows WSL2
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with ≥12 GB VRAM (RTX 3090, 4090, A5000, A6000)
- **CUDA**: 11.2+ recommended
- **Disk**: ~150 GB for full dataset + weights

---

## Dataset

### OncoLung60K

The OncoLung60K dataset contains 60,000 H&E-stained patches from 65 patients across four classes:

| Class | Patches | Patients |
|-------|---------|----------|
| Adenocarcinoma | 15,000 | 16 |
| Squamous Cell Carcinoma | 15,000 | 21 |
| Small Cell Lung Cancer | 15,000 | 12 |
| Normal lung tissue | 15,000 | 16 |
| **Total** | **60,000** | **65** |

**Resolution**: 512×512 pixels at 20× magnification.
**Inter-rater agreement**: Cohen's κ = 0.87 (two pathologists, 6,000-image subset).
**License**: CC BY-NC 4.0 (non-commercial research use only).

**Download**:
```bash
python scripts/download_dataset.py --output data/
# Or directly from Zenodo: https://zenodo.org/records/14995223
```

See [docs/DATASET.md](docs/DATASET.md) for full specifications.

### LungHist700

External benchmark dataset by [Diosdado et al. 2024](https://doi.org/10.1038/s41597-024-03944-3). Download from the original repository.

---

## Usage

### Train a single model

```bash
python -m src.train \
    --config configs/oncolung_modified_convnext.yaml \
    --data_dir data/oncolung60k \
    --output_dir runs/exp1
```

### Run patient-wise 5-fold cross-validation

```bash
python -m src.kfold_cv \
    --config configs/oncolung_modified_convnext.yaml \
    --data_dir data/oncolung60k \
    --output_dir runs/cv_modified_convnext \
    --folds 5
```

### Benchmark all baselines

```bash
python -m src.benchmark \
    --configs_dir configs/ \
    --data_dir data/oncolung60k \
    --output_dir runs/benchmark
```

### Generate Grad-CAM visualizations

```bash
python -m src.utils.explainability \
    --weights runs/cv_modified_convnext/fold0/best.pt \
    --images data/sample_test_images/ \
    --output_dir gradcam_outputs/
```

### Statistical significance testing

```bash
python -m src.utils.metrics --statistical_tests \
    --runs runs/cv_*/per_fold_metrics.csv \
    --output statistical_tests.csv
```

---

## Reproducing Paper Results

To exactly reproduce the numbers in the paper:

```bash
# This reproduces Table 5 (SOTA comparison) and Table 7 (extended ablation)
bash scripts/reproduce_paper.sh
```

Expected runtime: ~5-7 days on a single NVIDIA RTX 6000 (24 GB).

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for a step-by-step guide.

### Headline numbers (from the paper)

| Dataset | Model | Accuracy | Macro F1 | ROC-AUC |
|---------|-------|----------|----------|---------|
| OncoLung60K | Modified ConvNeXt + ECB | **91.3 ± 0.9%** | **91.0 ± 1.0%** | **0.974 ± 0.006** |
| LungHist700 | Modified ConvNeXt + ECB | **96.8 ± 1.1%** | — | — |

Reported under patient-wise stratified 5-fold cross-validation; ± denotes std across folds.

---

## Pretrained Models

Pretrained weights for all 5 cross-validation folds are released on Zenodo (separate from the dataset to keep file sizes manageable):

```bash
# Download all 5 folds (~1.5 GB)
python scripts/download_weights.py --output checkpoints/
```

Or manually:
- Fold 0–4 weights: [Zenodo link](https://zenodo.org/uploads/20410103)

---

## Citation

If you use this code, model, or dataset, please cite:

```bibtex
@article{Ahmad2026OncoLung,
  author = {Ahmad, Mansoor and Raja, Gulistan},
  title = {Leveraging the Modified ConvNeXt Model and OncoLung60K Dataset for Lung Cancer Diagnosis},
  journal = {Journal of Intelligent \& Fuzzy Systems},
  year = {2026},
  note = {Manuscript ID: IFS-26-0249}
}

---

## License

- **Code**: [MIT License](LICENSE)
- **Dataset**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial research use only)
- **Pretrained weights**: CC BY-NC 4.0

---

## Acknowledgments

- Pathologist Muhammad Asif (National University of Medical Sciences, NUMS) for slide preparation, H&E staining, and annotation.
- Muhammad Shaban (postdoctoral fellow, Harvard Medical School) for guidance on histopathology research challenges.
- This work was partially sponsored by NVIDIA (USA) through the RTX 6000 grant under the Applied Research Program.
- IRB approval: NUMS-IRB-2023-021.

---

## Contact

- **Mansoor Ahmad**: mansoor.ahmad@students.uettaxila.edu.pk
- **Gulistan Raja**: gulistan.raja@uettaxila.edu.pk
- **Affiliation**: University of Engineering and Technology, Taxila, Pakistan

For questions about the code, please [open an issue](https://github.com/mansoor2k17/OncoLung60k/issues). For dataset-specific questions, please email the authors directly.
