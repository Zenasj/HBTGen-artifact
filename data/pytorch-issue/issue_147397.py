import torch.nn as nn

from dataclasses import dataclass
import torch 

@dataclass
class MyStaticInput:
    int_1: int 
    int_2: int

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, static):
        return x + static.int_1 + static.int_2

from torch.utils._pytree import register_constant 

register_constant(MyStaticInput)

torch.export.export(Foo(), (torch.randn(1), MyStaticInput(1, 2)), strict=False)