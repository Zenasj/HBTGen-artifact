# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.sin()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 3, 3, dtype=torch.float32)

