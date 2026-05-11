"""
Model builder for SOTA baseline comparisons.

Supports:
  - ResNet50, DenseNet121, EfficientNet-B0
  - ConvNeXt-Tiny/Small/Base
  - ViT-B/16, Swin-B
  - Modified ConvNeXt (ours)
"""
from typing import Optional

import torch.nn as nn

from src.models.modified_convnext import modified_convnext_base


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Build a model by name.

    Args:
        name: One of:
            - 'modified_convnext'         (ours)
            - 'convnext_tiny|small|base'
            - 'resnet50'
            - 'densenet121'
            - 'efficientnet_b0'
            - 'vit_base_patch16_224'
            - 'swin_base_patch4_window7_224'
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        **kwargs: Forwarded to the underlying constructor.

    Returns:
        An `nn.Module` ready for training.
    """
    name = name.lower()

    if name == "modified_convnext":
        return modified_convnext_base(num_classes=num_classes, **kwargs)

    # All baselines use timm
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for baselines. Install with `pip install timm`."
        ) from e

    timm_name_map = {
        "convnext_tiny":  "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base":  "convnext_base",
        "resnet50":       "resnet50",
        "densenet121":    "densenet121",
        "efficientnet_b0": "efficientnet_b0",
        "vit_b16":        "vit_base_patch16_224",
        "vit_base":       "vit_base_patch16_224",
        "vit_base_patch16_224": "vit_base_patch16_224",
        "swin_b":         "swin_base_patch4_window7_224",
        "swin_base":      "swin_base_patch4_window7_224",
        "swin_base_patch4_window7_224": "swin_base_patch4_window7_224",
    }
    if name not in timm_name_map:
        raise ValueError(
            f"Unknown model name '{name}'. Available: "
            f"{['modified_convnext'] + list(timm_name_map.keys())}"
        )

    return timm.create_model(
        timm_name_map[name],
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


def count_params(model: nn.Module) -> float:
    """Return the parameter count in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_flops(model: nn.Module, input_size=(1, 3, 224, 224)) -> float:
    """Return the FLOPs in GFLOPs (forward only).

    Requires `pip install fvcore`.
    """
    import torch
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    x = torch.randn(*input_size)
    flops = FlopCountAnalysis(model, x).total() / 1e9
    return flops
