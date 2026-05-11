# OncoLung60K Dataset Specification

## Overview

OncoLung60K is a balanced 60,000-image patch dataset for lung cancer subtype classification, curated from 65 patients across four diagnostic classes.

| Class | Patches | Patients | Description |
|-------|---------|----------|-------------|
| Adenocarcinoma (Adeno) | 15,000 | 22 | Glandular structures, mucin production |
| Squamous Cell Carcinoma (SCC) | 15,000 | 19 | Keratinisation, intercellular bridges |
| Small Cell Lung Cancer (SCLC) | 15,000 | 12 | Densely packed small cells, salt-and-pepper chromatin |
| Normal lung tissue | 15,000 | 12 | Intact alveolar architecture, no atypia |
| **Total** | **60,000** | **65** | — |

## Data Acquisition

### Patient Cohort

- 65 patients (42 male, 23 female)
- Age range: 32–87 years (mean 62.1 ± 11.4)
- Source: Pathology archives at the National University of Medical Sciences (NUMS), Pakistan
- Period: 2022–2024
- Inclusion criteria: Confirmed lung cancer diagnosis, no prior chemotherapy/radiation
- IRB approval: NUMS-IRB-2023-021

### Imaging Protocol

| Parameter | Value |
|-----------|-------|
| Microscope | Olympus CX23 |
| Objectives | 10× and 20× |
| Camera | 5MP USB-mounted CMOS |
| Native resolution | 2592 × 1944 pixels |
| Effective magnification (output) | 20× |
| Patch size | 512 × 512 pixels |
| Staining | Hematoxylin and eosin (H&E), 5–7 min protocol |
| Section thickness | 4 μm |

### Annotation

Two board-certified pathologists with ≥10 years of experience independently annotated each patch.

**Inter-rater reliability** (Cohen's κ on a stratified 6,000-image subset):

| Class | Cohen's κ | % Agreement |
|-------|-----------|-------------|
| Adenocarcinoma | 0.86 | 92.4% |
| Squamous Cell Carcinoma | 0.83 | 90.8% |
| Small Cell Lung Cancer | 0.89 | 94.2% |
| Normal | 0.91 | 95.6% |
| **Overall** | **0.87** | **93.3%** |

Disagreements (n=187) were adjudicated by a third senior pathologist, whose decision is taken as ground truth.

## Preprocessing Pipeline

```
Raw microscope image (RGB)
      ↓
RGB → HSV conversion
      ↓
Otsu threshold on Saturation channel
      ↓
Morphological closing (3×3 kernel, 2 iterations)
      ↓
Non-overlapping 512×512 tile extraction within tissue mask
      ↓
Manual exclusion of blood-stained, folded, out-of-focus regions (~1.4%)
      ↓
Final dataset: 60,000 patches
```

Implementation in `src/data/preprocessing.py`.

## File Structure

After downloading and extracting from Zenodo:

```
data/oncolung60k/
├── adeno/                       # 15,000 images
│   ├── patient_03_y00512_x01024.jpg
│   ├── patient_03_y01024_x01024.jpg
│   └── ...
├── scc/                         # 15,000 images
│   ├── patient_18_y00512_x00512.jpg
│   └── ...
├── sclc/                        # 15,000 images
└── normal/                      # 15,000 images
```

Each filename encodes:
- `patient_<ID>`: anonymised patient identifier
- `y<...>_x<...>`: tile coordinates within the source image

## Patient-Wise Cross-Validation Splits

The 5-fold patient-wise splits used in the paper are released in `splits/oncolung60k_5fold.csv`.

```csv
filepath,label,patient_id,fold
adeno/patient_03_y00512_x01024.jpg,0,P003,2
scc/patient_18_y01536_x02048.jpg,1,P018,4
...
```

**Verify no leakage**:
```bash
python scripts/verify_no_leakage.py --csv splits/oncolung60k_5fold.csv
```

## Download

The dataset is hosted on Zenodo (DOI: [10.5281/zenodo.14995223](https://zenodo.org/records/14995223)).

```bash
# Option 1: Helper script
python scripts/download_weights.py --output data/ --type dataset

# Option 2: Direct from Zenodo
wget https://zenodo.org/records/14995223/files/OncoLung60K_patches.tar.gz
tar -xzf OncoLung60K_patches.tar.gz -C data/
```

## License

The dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for **non-commercial research use only**.

When using the dataset, please cite:

```bibtex
@dataset{Ahmad2025OncoLung60K,
  author = {Ahmad, Mansoor and Raja, Gulistan},
  title = {OncoLung60K: A Large-Scale Histopathological Image Dataset for Lung Cancer Subtype Classification},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14995223},
  url = {https://zenodo.org/records/14995223}
}
```

## Limitations

The dataset is provided as a research resource with the following acknowledged limitations:

1. **Single-centre cohort**: All 65 patients are from a single institution (NUMS, Pakistan). Cross-population generalisation is not established.

2. **Microscope photography ≠ WSI**: Images are captured via a USB-mounted camera, not a calibrated whole-slide scanner. This introduces vignetting, focus drift, and exposure variability not present in clinical WSI pipelines.

3. **Patch-level only**: The dataset is provided as patches, not whole slides. Slide-level identifiers are not preserved.

4. **Class imbalance at patient level**: While balanced at the patch level (15K each), patient counts are imbalanced (Adeno: 22 vs SCLC: 12).

We recommend using OncoLung60K as a development dataset, not a clinical benchmark. Multi-centre validation on calibrated WSI scanners is required before any clinical claim.

## Comparison with Other Lung Histopathology Datasets

| Dataset | Patches | Classes | Patients | SCLC | License |
|---------|---------|---------|----------|------|---------|
| LC25000 | 25,000 | 5 (3 lung, 2 colon) | N/A | ❌ | CC BY 3.0 |
| LungHist700 | 691 | 7 | 45 | ❌ | CC BY 4.0 |
| **OncoLung60K (this work)** | **60,000** | **4** | **65** | **✓** | CC BY-NC 4.0 |

Key differentiator: OncoLung60K is the first balanced public dataset that includes Small Cell Lung Cancer (SCLC) at scale.
