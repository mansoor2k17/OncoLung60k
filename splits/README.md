# Patient-Wise Cross-Validation Splits

This directory contains the **exact CSV files** used for patient-wise stratified 5-fold cross-validation in the paper. Use these to reproduce results bit-exactly.

## Files

| File | Patches | Patients | Classes |
|------|---------|----------|---------|
| `oncolung60k_5fold.csv` | 60,000 | 65 | 4 (Adeno, SCC, SCLC, Normal) |
| `lunghist700_5fold.csv` | 691 | 45 | 7 (3 ACA + 3 SCC + Normal) |

## CSV Schema

All split files have the same columns:

```csv
filepath,label,patient_id,fold
adeno/patient_03_y00512_x01024.jpg,0,03,2
scc/patient_18_y01536_x02048.jpg,1,18,4
sclc/patient_42_y02560_x03584.jpg,2,42,0
normal/patient_07_y00512_x00512.jpg,3,07,1
```

- **filepath**: relative path to the image, under your data root
- **label**: integer class label (0..N-1)
- **patient_id**: anonymised patient identifier (string)
- **fold**: assigned fold number (0..4)

## Verifying No Patient Leakage

Run this command to confirm zero patient overlap between folds:

```bash
python scripts/verify_no_leakage.py --csv splits/oncolung60k_5fold.csv
```

Expected output:

```
Loaded 60000 rows from splits/oncolung60k_5fold.csv
Patients: 65
Folds:    [0, 1, 2, 3, 4]

--- Per-fold summary ---
 fold  n_patches  n_patients  class_0  class_1  class_2  class_3
    0      12000          13     3000     3000     3000     3000
    1      12000          13     3000     3000     3000     3000
    2      12000          13     3000     3000     3000     3000
    3      12000          13     3000     3000     3000     3000
    4      12000          13     3000     3000     3000     3000

--- Patient leakage check ---
OK: No patient leakage across 5 folds
```

## Re-Generating Splits

If you want to regenerate splits with a different seed, use:

```bash
python -m src.utils.splits \
    --input_csv splits/oncolung60k_master.csv \
    --output_csv splits/oncolung60k_5fold_seed99.csv \
    --n_splits 5 \
    --seed 99
```

⚠️ **WARNING**: Re-generating with a different seed will produce different folds. Use only when intentionally exploring alternatives. For paper reproduction, use the provided CSVs.

## Note on Patient ID Anonymisation

All `patient_id` values in these CSVs are anonymised. The mapping to original patient identifiers is held by the authors and is not released to comply with HIPAA-equivalent standards (IRB approval NUMS-IRB-2023-021).

## Generation Process (Provenance)

The splits were generated with:
- `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- Grouping variable: `patient_id`
- Stratification variable: `label`
- This guarantees zero patient overlap and balanced class proportions across folds.
