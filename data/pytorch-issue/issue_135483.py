# torch.rand(B, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Demonstrates bias argument accepting numeric values (truthy)
        self.linear = nn.Linear(in_features=3, out_features=5, bias=10)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches Linear layer's in_features=3 with batch dimension
    B = 1  # Example batch size
    return torch.rand(B, 3, dtype=torch.float)

