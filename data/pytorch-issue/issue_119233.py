# torch.rand(5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x  # Pass-through to allow testing storage properties

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, dtype=torch.float32)  # Matches example input shape and type

