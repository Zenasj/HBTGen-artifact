import torch.nn as nn

py
from typing import Callable, Optional

import torch

def test(x: torch.Tensor) -> torch.Tensor:
    return x

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a: Optional[Callable[[torch.Tensor], torch.Tensor]] = test

    def forward(self, x: torch.Tensor):
        if self.a is not None:
            return self.a(x)
        return x

model = A()
torch.jit.script(model)