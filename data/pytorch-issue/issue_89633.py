# torch.rand(3, 2, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.ops.aten._to_copy(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 2, 4)

