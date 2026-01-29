# torch.rand((), dtype=torch.float32)  # Scalar input as per nonzero() examples
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nonzero(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(())  # Returns a scalar tensor (0-dim)

