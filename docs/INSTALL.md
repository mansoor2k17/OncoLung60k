# Installation Guide

## System Requirements

- **OS**: Ubuntu 20.04 (recommended), macOS, or Windows WSL2
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA 11.2+ and ≥12 GB VRAM
- **Disk**: ~150 GB for dataset + model weights + experiment outputs
- **RAM**: 16 GB minimum (32 GB recommended for full training)

## Option 1: Pip (Recommended for Development)

```bash
# 1. Clone repository
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 3. Install PyTorch (visit https://pytorch.org for your CUDA version)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify installation
python scripts/smoke_test.py
```

Expected output: `ALL SMOKE TESTS PASSED`.

## Option 2: Docker (Recommended for Reproduction)

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k

# Build the image (takes ~10 minutes)
docker build -t oncolung60k -f docker/Dockerfile .

# Run interactively with GPU
docker run --gpus all -v $(pwd):/workspace -it oncolung60k

# Inside container:
python scripts/smoke_test.py
```

Or with docker-compose:

```bash
cd docker
docker-compose up -d
docker-compose exec oncolung60k bash
```

## Option 3: Conda

```bash
git clone https://github.com/mansoor2k17/OncoLung60k.git
cd OncoLung60k

conda create -n oncolung60k python=3.8 -y
conda activate oncolung60k

conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt

python scripts/smoke_test.py
```

## Verifying GPU Access

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

If `CUDA: False`, your installation will fall back to CPU (very slow).

## Common Issues

### Issue: `ImportError: libGL.so.1: cannot open shared object file`

**Solution**:
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Issue: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in your config:
```yaml
batch_size: 16  # or 8
```

### Issue: `ModuleNotFoundError: No module named 'pytorch_grad_cam'`

**Solution**: Install grad-cam (optional, only needed for explainability):
```bash
pip install grad-cam==1.4.6
```

### Issue: Mixed-precision training crashes

**Solution**: Disable AMP in your config:
```yaml
amp: false
```

## Next Steps

After successful installation, see:
- [DATASET.md](DATASET.md) for downloading OncoLung60K
- [REPRODUCE.md](REPRODUCE.md) for reproducing paper results
- [HYPERPARAMETERS.md](HYPERPARAMETERS.md) for tuning
