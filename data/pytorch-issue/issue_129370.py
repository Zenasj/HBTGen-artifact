py
import torch
from torch import Tensor
from typing import Tuple

@torch.library.custom_op("mylib::foo", mutates_args={})
def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
    return x * y[0] * y[1]