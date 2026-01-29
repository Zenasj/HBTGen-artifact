# torch.rand(1, 1, dtype=torch.float32)  # Inferred input shape from test script's usage
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.sin() + x.cos()

def my_model_function():
    # Returns the model instance matching the test script's function
    return MyModel()

def GetInput():
    # Generates a tensor matching the input expected by MyModel
    return torch.rand(1, 1, dtype=torch.float32)

