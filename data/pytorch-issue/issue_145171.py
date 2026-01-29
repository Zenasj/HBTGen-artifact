# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x):
        return x * self.value

def my_model_function():
    return MyModel(value=2)  # Matches example's value=2

def GetInput():
    return torch.rand(3)  # Matches example's input shape (3 elements)

