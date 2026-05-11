"""
Smoke test: Verify the codebase runs end-to-end in ~5 minutes.

Tests:
  1. Model construction.
  2. Forward pass.
  3. Single training step on synthetic data.
  4. Metric computation.

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

print("=" * 60)
print("Smoke test for OncoLung60K Modified ConvNeXt")
print("=" * 60)

# -------- 1. Model construction --------
print("\n[1/4] Building Modified ConvNeXt...")
from src.models.modified_convnext import modified_convnext_base
model = modified_convnext_base(num_classes=4, ls_init=0)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"      OK ({n_params:.1f}M parameters)")

# -------- 2. Forward pass --------
print("\n[2/4] Forward pass with random input...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Device: {device}")
model = model.to(device)
x = torch.randn(2, 3, 256, 256, device=device)
y = model(x)
assert y.shape == (2, 4), f"Expected (2, 4), got {y.shape}"
print(f"      OK  (output shape: {y.shape})")

# -------- 3. Single training step --------
print("\n[3/4] One training step on synthetic data...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 4, (2,), device=device)
loss = criterion(model(x), target)
loss.backward()
optimizer.step()
print(f"      OK  (loss: {loss.item():.4f})")

# -------- 4. Metrics --------
print("\n[4/4] Metric computation...")
from src.utils.metrics import compute_all_metrics
y_true = np.random.randint(0, 4, size=200)
probs = np.random.dirichlet([1, 1, 1, 1], size=200)
m = compute_all_metrics(y_true, probs, n_classes=4)
print(f"      OK  (accuracy: {m['accuracy']:.4f}, AUC: {m['auc_macro_ovr']:.4f})")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
