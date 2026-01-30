py
import torch
from torch import Tensor
from typing import Tuple

@torch.library.custom_op("mylib::bar", mutates_args={}, device_types="cpu")
def bar(device: torch.device) -> Tensor:
    return torch.ones(3)

bar(cpu)