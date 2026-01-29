# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, y):
        super().__init__()
        self.y = nn.Parameter(y)  # Fixed parameter, like in original issue's y

    def forward(self, x):
        return torch.cdist(x, self.y)

def my_model_function():
    # Initialize with random y of shape (2,3) as in the original issue's example
    y = torch.randn(2, 3)
    return MyModel(y)

def GetInput():
    # Returns a tensor matching the input x in the original issue's example
    return torch.randn(2, 3, requires_grad=True)

