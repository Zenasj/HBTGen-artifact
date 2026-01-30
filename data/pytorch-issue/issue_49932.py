import torch.nn as nn

import torch
from typing import List
from torch.fx.symbolic_trace import Tracer

class TestModule(torch.nn.Module):
    
    def forward(self, x: torch.Tensor) -> List[str]:
        s: List[str] = []
        return s
    
m = TestModule()
graph = Tracer().trace(m)

print(graph.python_code('self'))

import torch
import typing
def forward(self, x : torch.Tensor) -> typing.List[str]:
    return []

import torch
import typing
def forward(self, x : torch.Tensor):
    return []