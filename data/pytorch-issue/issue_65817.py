# torch.rand(3, 300, 300, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the original issue's torch.all operation with dim=1
        return torch.all(x, dim=1)

def my_model_function():
    # Returns the model instance with the problematic torch.all operation
    return MyModel()

def GetInput():
    # Generates a random integer tensor matching the original input shape and dtype
    return torch.randint(0, 2, (3, 300, 300), dtype=torch.int32)

