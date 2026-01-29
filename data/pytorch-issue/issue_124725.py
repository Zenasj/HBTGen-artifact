from dataclasses import dataclass, astuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

@dataclass
class D:
    s: torch.Tensor
    y: torch.Tensor

    def __iter__(self):
        return iter(astuple(self))

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @staticmethod
    def dataclass_func(x):
        s = x.mean()
        y = torch.ones_like(x)
        return D(s, y)
    
    @staticmethod
    def tuple_func(x):
        s = x.mean()
        y = torch.ones_like(x)
        return (s, y)
    
    def forward(self, x):
        dc_output = checkpoint(self.dataclass_func, x, use_reentrant=True)
        s_dc = dc_output.s

        tuple_output = checkpoint(self.tuple_func, x, use_reentrant=True)
        s_tuple = tuple_output[0]

        discrepancy = (s_dc.requires_grad != s_tuple.requires_grad)
        return torch.tensor(discrepancy, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(40, requires_grad=True)

