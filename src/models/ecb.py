"""
Enhanced ConvNeXt Block (ECB).

The ECB augments the standard ConvNeXt block with three components:
  1. Multi-Scale Pooling (MSP): Average and max pooling at scale 1
     produce complementary local statistics.
  2. Contrast Enhancement (CE): The element-wise difference of max and
     average pooled feature maps emphasises local intensity variations
     associated with cellular structures.
  3. Feature Fusion (FF): Pooled maps and the depthwise-conv output
     are concatenated channel-wise and fused via a 1x1 conv.

Each component can be toggled independently for ablation studies.

Reference:
    Ahmad & Raja, "Leveraging the Modified ConvNeXt Model and OncoLung60K
    Dataset for Lung Cancer Diagnosis," Journal of Intelligent & Fuzzy
    Systems, 2026 (under review).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm operating on (B, C, H, W) tensors with channel-first layout."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class EnhancedConvNeXtBlock(nn.Module):
    """Enhanced ConvNeXt Block with toggleable ECB components.

    Args:
        dim: Number of input/output channels.
        drop_path: Stochastic depth probability.
        use_msp: If True, enable Multi-Scale Pooling (avg + max).
        use_ce:  If True, enable Contrast Enhancement (max - avg path).
        use_ff:  If True, enable Feature Fusion (concat + 1x1 conv).
        ls_init: Layer-scale initialization (0 = disabled).
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        use_msp: bool = True,
        use_ce: bool = True,
        use_ff: bool = True,
        ls_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_msp = use_msp
        self.use_ce = use_ce
        self.use_ff = use_ff

        # Standard ConvNeXt path
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if ls_init > 0:
            self.gamma = nn.Parameter(ls_init * torch.ones((dim,)), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ECB components
        if self.use_msp or self.use_ce:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        if self.use_ff:
            # Compute number of input channels for fusion 1x1 conv
            ff_in_channels = dim  # base depthwise output
            if self.use_msp:
                ff_in_channels += 2 * dim  # avg + max channels
            if self.use_ce:
                ff_in_channels += dim  # difference channel

            self.fusion = nn.Sequential(
                nn.Conv2d(ff_in_channels, dim, kernel_size=1),
                LayerNorm(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            )
        else:
            self.fusion = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Standard depthwise convolution
        out = self.dwconv(x)

        # ECB enhancement path
        if self.fusion is not None:
            B, C, H, W = out.shape
            cat_feats = [out]

            if self.use_msp:
                p_avg = self.avg_pool(out).expand(-1, -1, H, W)
                p_max = self.max_pool(out).expand(-1, -1, H, W)
                cat_feats.extend([p_avg, p_max])

            if self.use_ce:
                if self.use_msp:
                    p_diff = p_max - p_avg
                else:
                    p_avg_local = self.avg_pool(out).expand(-1, -1, H, W)
                    p_max_local = self.max_pool(out).expand(-1, -1, H, W)
                    p_diff = p_max_local - p_avg_local
                cat_feats.append(p_diff)

            fused = torch.cat(cat_feats, dim=1)
            enhanced = self.fusion(fused)
            out = out + enhanced  # residual within ECB

        # MLP head (channel-last for efficiency)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm.weight * (out - out.mean(-1, keepdim=True)) / (
            (out - out.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + 1e-6
        ).sqrt() + self.norm.bias
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = out.permute(0, 3, 1, 2)  # back to (B, C, H, W)

        out = identity + self.drop_path(out)
        return out


class DropPath(nn.Module):
    """Stochastic depth, per-sample (Huang et al. 2016)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor.floor_()
        return x.div(keep_prob) * rand_tensor


if __name__ == "__main__":
    # Smoke test: forward pass with all components enabled
    block = EnhancedConvNeXtBlock(dim=128, use_msp=True, use_ce=True, use_ff=True)
    x = torch.randn(2, 128, 32, 32)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    assert y.shape == x.shape
    print("ECB forward pass: OK")
