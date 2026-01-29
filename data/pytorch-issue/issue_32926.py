# torch.rand(B, 3, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.expand((-1, 3, 3, 1))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, 1, 1)  # Matches original test input dimensions (B=3, C=3, H=1, W=1)

