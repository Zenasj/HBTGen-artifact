# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn
from torch import Tensor

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # Matches the original Linear(10,10) model
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

def my_model_function():
    return MyModel()  # Returns the core model from the issue's example

def GetInput():
    B = 256  # Matches batch size from the original reproduction script
    return torch.randn(B, 10)  # Matches input shape (B, D) where D=10

