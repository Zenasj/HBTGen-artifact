# torch.rand(1, 10, 1, 1, dtype=torch.float32)  # Input shape: B=1, C=10, H=1, W=1
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))
        self.register_parameter("weight1", weight)
        self.weight2 = self.weight1  # Explicitly tie parameters

    def forward(self, x):
        # Flatten input to match original 1D operation
        x = x.view(-1)
        t = torch.matmul(self.weight1, x) + torch.matmul(self.weight2, x)
        return t.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, 1, 1)  # Matches B=1, C=10, H=1, W=1

