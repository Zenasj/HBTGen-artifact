# torch.rand(10, 0, dtype=torch.float32)  # Input shape inferred from the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.aminmax(x, dim=0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 0, dtype=torch.float32, device='cuda')

