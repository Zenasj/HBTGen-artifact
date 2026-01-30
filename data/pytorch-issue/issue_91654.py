import torch.nn as nn

from typing import Tuple

import torch
from torch import nn


def forward_hook(_layer: nn.Module, _input: Tuple[torch.Tensor, ...], _output: torch.Tensor) -> torch.Tensor:
    ...
    return _output


nn.Module().register_forward_hook(forward_hook)