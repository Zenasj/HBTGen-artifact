# torch.rand(1, dtype=torch.bool)  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        v1 = torch.gt(x, x)
        v2 = v1.expand(1, 1)
        v3 = v2.add_(v2, alpha=1)
        return v3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([True])  # Matches the original input's shape and dtype

