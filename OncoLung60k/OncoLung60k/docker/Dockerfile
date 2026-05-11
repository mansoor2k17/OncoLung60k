# ============================================================
# Dockerfile for OncoLung60K Modified ConvNeXt
# ============================================================
# Provides a reproducible environment matching the paper's
# software stack: PyTorch 1.10, CUDA 11.2, timm 0.9, Python 3.8.
#
# Build:
#   docker build -t oncolung60k -f docker/Dockerfile .
#
# Run (with GPU):
#   docker run --gpus all -v $(pwd):/workspace -it oncolung60k
# ============================================================

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-distutils \
    git \
    wget \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Default python = 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 11.3 wheels (compatible with CUDA 11.2 driver)
RUN pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Workspace
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Smoke test on build (catches missing imports)
COPY src /workspace/src
COPY scripts /workspace/scripts
RUN python -c "from src.models.modified_convnext import modified_convnext_base; print('OK')"

CMD ["/bin/bash"]
