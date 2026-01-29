# torch.rand(B, C, dtype=torch.float32)  # Input shape inferred from original code example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nn.functional.glu(x, dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024, 512, dtype=torch.float32)

