# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the hasattr-based condition causing Dynamo compilation failure
        if hasattr(x, "attr"):
            return x + 1
        else:
            return x - 1

def my_model_function():
    # Returns the problematic model instance
    return MyModel()

def GetInput():
    # Creates input tensor with custom attribute to trigger hasattr check
    x = torch.rand(1)
    x.attr = False  # Matches the original issue's input configuration
    return x

