# torch.rand(1, 1, 1, 1, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, eps=0.9):
        super().__init__()
        self.eps = eps  # Fixed epsilon value as in the example

    def forward(self, x):
        return torch.logit(x, eps=self.eps)

def my_model_function():
    # Returns the model initialized with eps=0.9 as in the issue's test case
    return MyModel(eps=0.9)

def GetInput():
    # Returns a random tensor matching the input shape and dtype
    return torch.rand(1, 1, 1, 1, dtype=torch.float64)

