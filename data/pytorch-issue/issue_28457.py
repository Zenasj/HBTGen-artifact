# torch.rand(B, N, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.T

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3)  # Example input with shape (batch=2, features=3)

