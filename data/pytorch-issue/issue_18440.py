# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        for i in range(1, 10):
            x += float(i)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

