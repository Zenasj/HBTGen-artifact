# torch.rand((), dtype=torch.float32)  # Scalar input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.log2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)

