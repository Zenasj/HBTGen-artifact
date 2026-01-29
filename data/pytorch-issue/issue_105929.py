# torch.rand(1, dtype=torch.float32)  # Inferred input shape is (1,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

