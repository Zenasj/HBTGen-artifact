# torch.rand(1, 1000000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicate the in-place modifications causing the Inductor bug
        x[:, 20:40] += 1  # First slice modification
        x[:, 2:900025] = x[:, 1:900024] + 2  # Second larger slice modification
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1000000, dtype=torch.float32)

