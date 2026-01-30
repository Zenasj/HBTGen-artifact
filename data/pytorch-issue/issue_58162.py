import torch.nn as nn

import torch
import torch.fx
from typing import Tuple, List

class Foo(torch.nn.Module):
    def forward(self, x : Tuple[int]):
        return x[0]

traced = torch.fx.symbolic_trace(Foo())
scripted = torch.jit.script(traced)

import torch
import torch.fx
from typing import Tuple, List

class Foo(torch.nn.Module):
    def forward(self, x : Tuple[()]):
        return x[0]

traced = torch.fx.symbolic_trace(Foo())
print(traced.code)
scripted = torch.jit.script(traced)