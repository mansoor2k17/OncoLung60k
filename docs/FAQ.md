# Frequently Asked Questions

## Dataset

### Q: Why is OncoLung60K released under CC BY-NC 4.0 instead of a permissive license?

**A:** Medical imaging datasets are commonly released for non-commercial use only because:
1. The IRB approval (NUMS-IRB-2023-021) restricts use to research purposes.
2. Commercial use of patient-derived data requires additional consent procedures.

For commercial licensing inquiries, please email the corresponding author.

### Q: Can I use OncoLung60K to train a clinical product?

**A:** No. The dataset is for research only. Clinical deployment requires:
- Multi-centre validation
- Data from calibrated WSI scanners
- Regulatory approval (FDA SaMD or EU AI Act conformity)
- Quality management under ISO 13485

### Q: Why are there only 65 patients?

**A:** This is acknowledged as a primary limitation in the paper (Section 6). The dataset was curated from a single institution's pathology archive over two years. Multi-centre expansion is planned future work.

### Q: Why does class distribution differ at the patient vs patch level?

**A:** The dataset is balanced at the **patch** level (15K each) but not at the **patient** level (Adeno: 22 patients, SCC: 19, SCLC: 12, Normal: 12). This is because we selected approximately equal numbers of patches per class, but the available patient cohort had different numbers of confirmed cases per subtype.

### Q: How were patches extracted from the slides?

**A:** Through:
1. RGB → HSV conversion
2. Otsu thresholding on the saturation channel for tissue masking
3. Morphological closing
4. Non-overlapping 512×512 tile extraction at 20× effective magnification

Implementation in `src/data/preprocessing.py`.

### Q: Are the images WSI?

**A:** **No.** Images were captured through a USB-mounted camera attached to an Olympus CX23 microscope. This differs from calibrated whole-slide imaging (WSI) scanners. We acknowledge this in the paper's Limitations section.

## Methodology

### Q: Why patient-wise k-fold and not random splits?

**A:** Histopathology datasets contain many patches per patient. Random patch-level splits cause patches from the same patient to appear in both training and test sets. The model learns patient-specific features (e.g., staining variation) and accuracy is dramatically inflated.

Patient-wise splits guarantee zero overlap and produce honest performance estimates.

### Q: Why ConvNeXt and not a transformer?

**A:** Three reasons:
1. ConvNeXt achieves competitive performance with lower memory footprint than ViT/Swin
2. CNN-based explainability tools (Grad-CAM++) are well-established
3. CNNs are easier to deploy in clinical environments

We compare against ViT-B/16 and Swin-B in the paper.

### Q: What's the difference between MSP, CE, and FF?

**A:** Three components of the Enhanced ConvNeXt Block (ECB):
- **MSP (Multi-Scale Pooling)**: Average and max pooling produce complementary spatial statistics.
- **CE (Contrast Enhancement)**: The element-wise difference (max - avg) emphasises local intensity variations.
- **FF (Feature Fusion)**: Pooled maps are concatenated and fused via 1×1 conv.

Ablation in paper Table 7 isolates each component's contribution.

### Q: Is the ECB novel?

**A:** Honestly, no — the individual components are established in prior work (CBAM, spatial pyramid pooling, etc.). Our contribution is empirical: we identify the specific combination that addresses lung histopathology challenges. See Section 3.5 ("Positioning of the Contribution") for explicit framing.

## Reproducibility

### Q: Will my numbers exactly match the paper?

**A:** Not exactly, but within ±0.5% on accuracy. Sources of non-determinism:
- CUDA `cudnn.benchmark` mode
- Different GPU architectures
- Multi-threaded data loading

Set `torch.backends.cudnn.deterministic = True` for closer reproduction (slower).

### Q: Why do I get out-of-memory errors?

**A:** Reduce batch size in your config:
```yaml
batch_size: 16  # default 32; try 8 if still OOM
```

### Q: Training is too slow on my hardware. Can I skip some experiments?

**A:** Recommended priorities:
1. Modified ConvNeXt + 5-fold CV (essential)
2. Comparison with at least Swin-B (toughest competitor)
3. Ablation: Base vs Full ECB (essential)
4. Other baselines (optional, can be cited from paper)

### Q: Can I use a different optimiser?

**A:** Yes, edit your config:
```yaml
# Custom optimiser would require code modification
# in src/train.py — currently we hardcode AdamW
```

A future version may support optimiser selection via config.

## Clinical / Regulatory

### Q: Is this model FDA-approved?

**A:** **No.** This is a research prototype. No regulatory clearance has been sought.

### Q: Can the model replace a pathologist?

**A:** **No.** The model is designed to *support* pathologists in research settings, not replace them. Clinical deployment would require:
- Prospective trials
- Multi-centre validation
- IEC 62304 conformity
- ISO 13485 quality management
- Ongoing post-market surveillance

### Q: What about bias?

**A:** All 65 patients are from Pakistan. The model may not generalise to populations with:
- Different cancer prevalence patterns
- Different staining protocols
- Different scanner equipment

We acknowledge this limitation and plan multi-centre validation in future work.

## Code

### Q: Why do you use timm for baselines?

**A:** timm provides:
- Standardised model implementations
- Verified ImageNet-pretrained weights
- Consistent interface across architectures
- Active maintenance

This makes our comparisons fair and reproducible.

### Q: Can I use this code with PyTorch 2.x?

**A:** Most likely yes, but we test with PyTorch 1.10. PyTorch 2.x users may see warnings about deprecated APIs but functionality should work.

### Q: How do I contribute?

**A:** We welcome contributions:
1. Open an issue describing your proposed change
2. Fork the repo, create a feature branch
3. Submit a pull request with tests

Areas of particular interest:
- Foundation model integration (UNI, CONCH, Virchow)
- Federated learning extensions
- Slide-level inference pipelines
- Explainability methods beyond Grad-CAM

## Other

### Q: How do I cite this work?

**A:** See [README.md](../README.md) Citation section. Both the paper and dataset have separate citations.

### Q: I have a question not answered here. How do I get help?

**A:**
- **Code questions**: [Open a GitHub issue](https://github.com/mansoor2k17/OncoLung60k/issues)
- **Dataset questions**: Email the corresponding author (gulistan.raja@uettaxila.edu.pk)
- **Collaboration inquiries**: Email the first author (mansoor.ahmad@students.uettaxila.edu.pk)
