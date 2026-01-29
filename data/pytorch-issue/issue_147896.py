# torch.rand(3, 4, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x_complex = x.to(torch.complex64)
        return x_complex[:, :2]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4, 1, 1)

