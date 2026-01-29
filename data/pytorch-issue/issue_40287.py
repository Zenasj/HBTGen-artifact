# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Compute standard deviation along dimension 0 (matches the issue's expected behavior)
        return x.std(0)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape (2x3 from the issue example)
    return torch.rand(2, 3, dtype=torch.float32)

