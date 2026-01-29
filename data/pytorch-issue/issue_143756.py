# torch.rand(1, dtype=torch.float32)  # Input shape is a single-element tensor
import torch
from torch import nn

class Something:
    def __init__(self) -> None:
        self.__dict__["something"] = 'whatever'

class MyModel(nn.Module):
    def forward(self, x) -> torch.Tensor:
        Something()  # Triggers the graph break
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1)  # Matches the minimal input shape required

