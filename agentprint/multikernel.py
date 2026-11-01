"""Shared same-slot multikernel unit and the replacement RF integration."""
import math
from typing import Any

import torch
from torch import nn

from .config import ConfigError
from .model import RF


class MultiKernelUnit(nn.Module):
    """Map [N, in_channels, 2, T] to [N, out_channels, 2, T].

    Branches observe 3/5/7 original time slots with 22/21/21 channels.
    The projection maps their concatenated 64 channels to ``out_channels``:
    replacement uses 64; prepend can use the original 2 or 10 input channels.
    Every convolution has a bias, matching RF, and initializes with RF's
    Conv2d/BatchNorm2d rules. There is no residual or time-axis cropping.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if type(in_channels) is not int or in_channels not in (2, 10):
            raise ValueError('input channels must be 2 or 10')
        if type(out_channels) is not int or out_channels not in (2, 10, 64):
            raise ValueError('multikernel output channels must be 2, 10, or 64')
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, channels, (3, kernel),
                          stride=1, padding=(1, kernel // 2)),
                nn.BatchNorm2d(channels), nn.ReLU())
            for kernel, channels in ((3, 22), (5, 21), (7, 21))
        ])
        self.projection = nn.Sequential(nn.Conv2d(64, out_channels, 1),
                                        nn.BatchNorm2d(out_channels), nn.ReLU())
        # Match RF._initialize_weights without touching any Conv1d defaults.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                scale = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2. / scale))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([branch(values) for branch in self.branches], dim=1))


def build_replacement_model(params: dict[str, Any], in_channels: int = 2) -> RF:
    """Replace the entire first RF Conv/BN/ReLU at the approved width 2."""
    from .tuning import build_model
    if params['width'] != 2:
        raise ConfigError('replace model requires width=2')
    model = build_model(params, in_channels=in_channels)
    # Retain all later RF layers, pooling, reshape, classifier and dropouts.
    model.first_layer = nn.Sequential(MultiKernelUnit(in_channels, 64),
                                      *list(model.first_layer.children())[3:])
    return model
