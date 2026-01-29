# torch.rand(3, 4)  # Inferred input shape from example_inputs
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(3, 4))
        # Original p2 used CustomTensorPlainOut (two tensors), here replaced with standard tensor for reproducibility
        self.p2 = nn.Parameter(torch.ones(3, 4))

    def forward(self, x):
        a = (2 * self.p1 + self.p2).sum()
        return x + a

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4)

