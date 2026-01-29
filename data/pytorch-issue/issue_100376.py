# torch.rand(B, 1, 20, 20, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        for _ in range(3):
            x = torch.sin(x)
        x = torch.relu(x)
        for _ in range(3):
            x = torch.cos(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 20, 20, dtype=torch.float32)

