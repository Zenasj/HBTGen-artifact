# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, from_val, to_val):
        super().__init__()
        self.params = {"from": from_val, "to": to_val}  # Stores parameters with "from" key

    def forward(self, x):
        # Triggers the error when compiled due to "from" keyword in keyword arguments
        return x.random_(**self.params)

def my_model_function():
    # Initialized with parameters from the original issue's example
    return MyModel(from_val=-10, to_val=10)

def GetInput():
    # Matches the input shape used in the minified repro
    return torch.randn(2, 3)

