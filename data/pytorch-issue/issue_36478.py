# torch.rand(2, 3, dtype=torch.float32)  # Input is a tuple containing a tensor of shape (2, 3)
import torch
from torch import nn
from typing import Tuple

class MyModel(nn.Module):
    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        return x[0]

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 3), )

