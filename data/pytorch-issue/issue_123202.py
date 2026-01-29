# torch.rand(5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 5))  # Matches test's weight shape (10,5)
        self.bias = nn.Parameter(torch.randn(10))       # Matches test's bias shape (10)

    def forward(self, x):
        # Compute loss as per test's closure
        return (self.weight.mv(x) + self.bias).pow(2).sum()

def my_model_function():
    # Returns model instance with parameters initialized via __init__
    return MyModel()

def GetInput():
    # Returns input tensor matching test's inpt (shape 5)
    return torch.randn(5, dtype=torch.float32)

