# torch.rand(3)  # Inferred input shape (1D tensor of size 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        xs = [x]  # Wrap tensor in list to trigger ListVariable
        if hasattr(xs, 'foo'):
            return xs[0] + 1
        else:
            return xs[0] * 2

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random 1D tensor matching the input expected by MyModel
    return torch.rand(3)

