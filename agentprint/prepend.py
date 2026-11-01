"""Independent multikernel frontend before the complete RF classifier."""
from typing import Any

from torch import nn

from .config import ConfigError
from .model import RF
from .multikernel import MultiKernelUnit


def build_prepend_model(params: dict[str, Any], in_channels: int = 2) -> RF:
    """Apply RF(F(x)) with no residual, retaining every original RF layer."""
    from .tuning import build_model
    if params['width'] != 2:
        raise ConfigError('prepend model requires width=2')
    model = build_model(params, in_channels=in_channels)
    model.first_layer = nn.Sequential(MultiKernelUnit(in_channels, in_channels),
                                      model.first_layer)
    return model
