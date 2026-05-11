"""
Modified ConvNeXt architecture for lung cancer histopathology classification.

This implements the architecture described in:
    Ahmad & Raja, "Leveraging the Modified ConvNeXt Model and OncoLung60K
    Dataset for Lung Cancer Diagnosis," Journal of Intelligent & Fuzzy
    Systems, 2026 (under review).

The model follows the standard ConvNeXt-Base layout with the standard
ConvNeXt blocks replaced by Enhanced ConvNeXt Blocks (ECB).

Stage configuration (ConvNeXt-Base):
    - Stem: 4x4 conv, stride 4
    - Stage 1: 3 blocks, dim=128
    - Stage 2: 3 blocks, dim=256
    - Stage 3: 27 blocks, dim=512  (paper uses 9 for the modified variant)
    - Stage 4: 3 blocks, dim=1024
    - Head: GAP + Linear

Reference: Liu et al., "A ConvNet for the 2020s," CVPR 2022.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from src.models.ecb import EnhancedConvNeXtBlock, LayerNorm


class ModifiedConvNeXt(nn.Module):
    """Modified ConvNeXt with Enhanced ConvNeXt Blocks (ECB).

    Args:
        in_channels: Input image channels (default 3 for RGB).
        num_classes: Number of output classes (default 4 for OncoLung60K).
        depths: Number of ECB blocks per stage. Default [3, 3, 9, 3].
        dims: Channel dimensions per stage. Default [128, 256, 512, 1024].
        drop_path_rate: Maximum stochastic depth rate (linear decay).
        use_msp: If True, ECB uses multi-scale pooling.
        use_ce:  If True, ECB uses contrast enhancement.
        use_ff:  If True, ECB uses feature fusion.
        head_init_scale: Initial scale for the classifier weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (128, 256, 512, 1024),
        drop_path_rate: float = 0.1,
        use_msp: bool = True,
        use_ce: bool = True,
        use_ff: bool = True,
        head_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        # Stem (4x4 conv, stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0]),
        )

        # Downsampling layers between stages
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # ECB stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    EnhancedConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        use_msp=use_msp,
                        use_ce=use_ce,
                        use_ff=use_ff,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Classification head
        self.norm = LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        # Initialize
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages[0](x)
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](x)
        return x  # (B, C=1024, H/32, W/32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # Global average pool
        x = x.mean(dim=[-2, -1])  # (B, C)
        x = self.norm.weight * (x - x.mean(-1, keepdim=True)) / (
            (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + 1e-6
        ).sqrt() + self.norm.bias
        x = self.head(x)
        return x


def modified_convnext_base(
    num_classes: int = 4,
    pretrained: Optional[str] = None,
    **kwargs,
) -> ModifiedConvNeXt:
    """Modified ConvNeXt-Base configuration.

    Args:
        num_classes: Number of output classes.
        pretrained: Path to ConvNeXt-Base checkpoint to load (non-ECB
            weights only; ECB layers are randomly initialized).
        **kwargs: Forwarded to ModifiedConvNeXt constructor.
    """
    model = ModifiedConvNeXt(
        depths=(3, 3, 9, 3),
        dims=(128, 256, 512, 1024),
        num_classes=num_classes,
        **kwargs,
    )
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        if "model" in state:
            state = state["model"]
        if "state_dict" in state:
            state = state["state_dict"]
        # Load only matching keys (skip ECB-specific layers)
        msg = model.load_state_dict(state, strict=False)
        print(f"[modified_convnext_base] Loaded pretrained from {pretrained}")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    return model


if __name__ == "__main__":
    # Smoke test
    model = modified_convnext_base(num_classes=4)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {n_params:.1f}M")
    assert y.shape == (2, 4)
    print("ModifiedConvNeXt forward pass: OK")
