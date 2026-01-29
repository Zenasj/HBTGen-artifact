# torch.rand(B, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight, alpha=1.0, beta=0.1)

def my_model_function():
    weight = torch.randn(64, 64)
    bias = torch.randn(64)
    return MyModel(weight, bias)

def GetInput():
    return torch.rand(1, 64, dtype=torch.float32)

