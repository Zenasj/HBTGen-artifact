import torch.nn as nn

import torch
from enum import Enum

class MyEnum(Enum):
    A = "a"

class SomeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x[MyEnum.A])

x = {MyEnum.A: torch.rand(100, 1)}
model = torch.compile(SomeModel())
model(x)
model(x)