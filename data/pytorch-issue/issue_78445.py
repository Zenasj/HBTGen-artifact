# torch.rand(1, 2, 2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)  # Reproduces the permute operation causing discrepancies

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 2, 3, dtype=torch.float32)

