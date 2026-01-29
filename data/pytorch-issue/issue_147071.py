# torch.rand(875, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        var_17 = torch.slice_copy(input, dim=0, start=449, end=None, step=9223372036854775807)
        return torch.reciprocal(var_17)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(875, dtype=torch.float32)

