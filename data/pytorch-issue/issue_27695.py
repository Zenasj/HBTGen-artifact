# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

