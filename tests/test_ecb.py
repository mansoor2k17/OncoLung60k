"""Unit tests for the Enhanced ConvNeXt Block."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.models.ecb import EnhancedConvNeXtBlock


def test_full_ecb_forward():
    block = EnhancedConvNeXtBlock(dim=64, use_msp=True, use_ce=True, use_ff=True)
    x = torch.randn(2, 64, 16, 16)
    y = block(x)
    assert y.shape == x.shape


def test_base_only_forward():
    block = EnhancedConvNeXtBlock(dim=64, use_msp=False, use_ce=False, use_ff=False)
    x = torch.randn(2, 64, 16, 16)
    y = block(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("msp,ce,ff", [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_all_ablation_combinations(msp, ce, ff):
    block = EnhancedConvNeXtBlock(dim=32, use_msp=msp, use_ce=ce, use_ff=ff)
    x = torch.randn(2, 32, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_gradient_flow():
    block = EnhancedConvNeXtBlock(dim=64)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    y = block(x).sum()
    y.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_param_count_reasonable():
    """Ensure ECB doesn't add ridiculous numbers of parameters."""
    base = EnhancedConvNeXtBlock(dim=128, use_msp=False, use_ce=False, use_ff=False)
    full = EnhancedConvNeXtBlock(dim=128, use_msp=True, use_ce=True, use_ff=True)
    base_params = sum(p.numel() for p in base.parameters())
    full_params = sum(p.numel() for p in full.parameters())
    # ECB adds 1x1 fusion conv on (4*dim) channels -> dim, plus a 3x3 dwconv
    overhead = full_params - base_params
    assert overhead < base_params, "ECB overhead should be less than base block"
