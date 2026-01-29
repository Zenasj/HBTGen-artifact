# torch.rand(3, 3, 3, dtype=torch.float32)  # Inferred input shape from test parameters
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t + t

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, 3)  # Matches the input shape used in the test case

