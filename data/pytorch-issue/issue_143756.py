import torch.nn as nn

import torch


class Something:
    def __init__(self) -> None:
        self.__dict__["something"] = 'whatever'


class MyModule(torch.nn.Module):
    def forward(self, x) -> torch.Tensor:
        Something()
        return x

mod = torch.compile(MyModule())
mod(torch.randn(1))