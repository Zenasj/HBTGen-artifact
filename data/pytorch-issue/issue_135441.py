# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # In-place operation causing versioning issues when compiled
        x[...] = 0
        return x

def my_model_function():
    # Returns a model that performs in-place tensor mutation
    return MyModel()

def GetInput():
    # Returns a 1D tensor of shape (4,) with random values
    return torch.rand(4, dtype=torch.float32)

