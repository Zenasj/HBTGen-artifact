# torch.randint(10, 50, (4, 5), dtype=torch.int)  # Inferred input shape and dtype

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, scalar=7, alpha=3):
        super().__init__()
        self.scalar = scalar
        self.alpha = alpha

    def forward(self, x):
        return torch.add(x, self.scalar, alpha=self.alpha)

def my_model_function():
    return MyModel()  # Uses default scalar=7, alpha=3 from the issue's example

def GetInput():
    return torch.randint(low=10, high=50, size=(4,5), dtype=torch.int)

