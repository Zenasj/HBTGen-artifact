# torch.rand(B, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x[..., 1:] + 2  # Matches the original function's logic

def my_model_function():
    return MyModel()  # Returns the model instance

def GetInput():
    return torch.rand(1, 5, dtype=torch.float32)  # Matches input shape from the issue's example

