import torch.nn as nn

from typing import Any
from torch import nn


class Foo(nn.Module):
    def forward(self, *input: Any, **kwargs: Any) -> Any:
        pass

import torch


class Bar(Foo):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

import torch
from torch import nn


class Foo(nn.Module[torch.Tensor]):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

def f(x: Module):
   return x("hi")

x = nn.Linear(...)
f(x)

import torch
from torch import nn

class Foo(nn.Module):
    pass

class Bar(Foo):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tensor(3.5)