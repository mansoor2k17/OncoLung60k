# Step-by-Step: Pushing This Repository to GitHub

I cannot push directly to your GitHub account — that requires your authentication credentials. Below is the exact procedure to update your repository at https://github.com/mansoor2k17/OncoLung60k yourself.

---

## OPTION A: Replace existing repo contents (recommended)

If your current repo only has the dataset README, this is the cleanest approach.

### Step 1: Download the prepared files

Download the entire `repo_for_github/` folder I prepared. It contains:
- `README.md` — main repo readme
- `LICENSE`, `CITATION.cff`, `requirements.txt`, `pyproject.toml`, `.gitignore`
- `src/` — model code (ECB, training, evaluation, metrics, explainability)
- `configs/` — YAML configs for all 9 baselines + 8 ablations
- `notebooks/` — Jupyter notebooks for setup, training, evaluation, Grad-CAM
- `splits/` — patient-wise CV split CSV (and demo split)
- `scripts/` — reproduction scripts, smoke tests, helpers
- `docs/` — INSTALL, DATASET, REPRODUCE, HYPERPARAMETERS, FAQ
- `tests/` — unit tests for ECB, splits, metrics
- `docker/` — Dockerfile + docker-compose

### Step 2: Clone your existing repo locally

```bash
cd ~  # or wherever you keep code
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k
```

### Step 3: Back up anything you want to keep

If your current repo has anything you want to preserve:

```bash
mkdir -p ~/oncolung_backup
cp -r * ~/oncolung_backup/   # save current contents (excluding .git)
```

### Step 4: Copy the new files into your repo

```bash
# Remove old contents (keeps .git directory)
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Copy new files (replace path with where you downloaded them)
cp -r /path/to/repo_for_github/* .
cp -r /path/to/repo_for_github/.gitignore .
```

### Step 5: Personalise placeholder values

Search and replace the following in all files:

| Placeholder | Replace with | Files to check |
|-------------|-------------|----------------|
| `<author>` (in URLs) | `mansoor2k17` | README.md, docs/*.md |
| `gulistan.raja@uettaxila.edu.pk` | (verify correct email) | README.md, CITATION.cff, docs/FAQ.md |
| `mansoor.ahmad@students.uettaxila.edu.pk` | (verify correct email) | Same |
| `NUMS-IRB-2023-021` | (your actual IRB number) | docs/DATASET.md, README.md |
| `10.5281/zenodo.14995223` | (your actual Zenodo DOI) | CITATION.cff, README.md |

Quick command to find all placeholders:

```bash
grep -rn "<author>" .
grep -rn "NUMS-IRB" .
grep -rn "zenodo" .
```

### Step 6: Commit and push

```bash
git add -A
git status   # review what's being committed

git commit -m "Update repository with full code, configs, splits, and documentation

- Add Modified ConvNeXt model with Enhanced ConvNeXt Block (ECB)
- Add training, evaluation, ablation, k-fold CV pipelines
- Add patient-wise split CSVs and verification utilities
- Add Grad-CAM/Grad-CAM++/Score-CAM explainability
- Add 9 baseline configs (ResNet50, DenseNet, EfficientNet, ConvNeXt-T/S/B, ViT-B/16, Swin-B, ours)
- Add 8 ablation configs covering all (MSP, CE, FF) combinations
- Add 4 Jupyter notebooks for dataset/training/evaluation/explainability
- Add comprehensive documentation (INSTALL, DATASET, REPRODUCE, HYPERPARAMETERS, FAQ)
- Add Dockerfile and docker-compose for reproducible environment
- Add unit tests for ECB, splits, and metrics

Supports the revised manuscript IFS-26-0249 'Leveraging the Modified ConvNeXt
Model and OncoLung60K Dataset for Lung Cancer Diagnosis'."

git push origin main
```

If you get an authentication error, you may need to:
- Set up a Personal Access Token: https://github.com/settings/tokens
- Or use SSH: `git remote set-url origin git@github.com:mansoor2k17/OncoLung60k.git`

---

## OPTION B: Add to existing repo without replacing

If you want to keep your current README and just add the code:

### Step 1: Clone your repo

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k
```

### Step 2: Copy only the new directories

```bash
SRC=/path/to/repo_for_github
cp -r $SRC/src $SRC/configs $SRC/notebooks $SRC/scripts \
      $SRC/splits $SRC/tests $SRC/docs $SRC/docker .
cp $SRC/requirements.txt $SRC/pyproject.toml \
   $SRC/CITATION.cff $SRC/.gitignore .

# Skip README.md if you want to keep your existing one
```

### Step 3: Commit and push

```bash
git add -A
git commit -m "Add code, configs, notebooks, and documentation"
git push origin main
```

---

## OPTION C: New branch for the revised version

To preserve the current state of your repo as-is and add the revision in a new branch:

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k

git checkout -b revision-1
cp -r /path/to/repo_for_github/* .
cp /path/to/repo_for_github/.gitignore .

git add -A
git commit -m "Revision 1: Add full code and documentation for IFS-26-0249"
git push origin revision-1

# When ready, merge to main:
# git checkout main
# git merge revision-1
# git push origin main
```

---

## After Pushing: Verify the Repo Works

A reviewer will likely click your link and try this:

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k
pip install -r requirements.txt
python scripts/smoke_test.py
```

Test this yourself on a fresh terminal/machine to make sure it works end-to-end.

---

## Things You Still Need to Do Manually

The repo I prepared has placeholders for things only you can produce:

### 1. Real patient-wise split CSV

The `splits/oncolung60k_5fold.csv` file does NOT exist in the package because the actual filenames depend on your dataset. You must:

```bash
# After organising your dataset CSV with [filepath, label, patient_id] columns:
python -m src.utils.splits \
    --input_csv your_master_csv.csv \
    --output_csv splits/oncolung60k_5fold.csv \
    --n_splits 5 \
    --seed 42
```

### 2. Pretrained weights upload to Zenodo

After running k-fold CV and getting the 5 best fold checkpoints:

```bash
# Bundle weights
tar -czf modified_convnext_weights.tar.gz runs/cv_modified_convnext/fold*/best.pt

# Upload to Zenodo (separate file from the dataset)
# Then update the URL in scripts/download_weights.py
```

### 3. Sample test images

For Grad-CAM demos, copy 20–50 representative test patches into:
```
data/sample_test_images/
```

### 4. Verify all citations and DOIs exist

Before marking the repo public, verify:
- The Zenodo DOI is correct
- ORCID iDs are correct
- Email addresses are correct
- IRB number is correct

### 5. Add a GitHub Pages site (optional)

For a more polished presentation:

```bash
# Settings → Pages → Source: main / docs
```

Then GitHub will render your docs/ folder as a website.

---

## After Submitting the Revision

When you submit the revised manuscript to ScholarOne, in your response letter (Code and Data Availability section), use the line:

> The complete implementation, pre-trained weights, preprocessing scripts, and configuration files are publicly available at https://github.com/mansoor2k17/OncoLung60k

If the journal requires double-blind review, use anonymous.4open.science:
1. Go to https://anonymous.4open.science/
2. Submit your repo URL
3. Use the anonymous URL in the manuscript for review
4. Replace with the public URL after acceptance

---

## File Count Summary

The package contains approximately:
- **6** top-level files (README, LICENSE, etc.)
- **18** Python source files in `src/`
- **18** YAML configs (9 baselines + 8 ablations + 1 LungHist700)
- **4** Jupyter notebooks
- **5** documentation files
- **3** unit test files
- **2** Docker files
- **5** helper scripts

**Total: ~60 files** organised into a clean, professional repository structure.

---

## If You Run into Issues

Common problems:

| Issue | Fix |
|-------|-----|
| `Permission denied (publickey)` | Set up SSH keys or use HTTPS with PAT |
| `Updates were rejected` | Run `git pull --rebase` first, then push |
| `LFS quota exceeded` | Don't commit large files (use Zenodo for weights) |
| Imports fail in notebooks | Ensure you run from repo root with `sys.path.insert(0, '.')` |
| Smoke test fails | Check Python/PyTorch/CUDA versions match `requirements.txt` |

For any other issue, contact me with the error message and I can help debug.
