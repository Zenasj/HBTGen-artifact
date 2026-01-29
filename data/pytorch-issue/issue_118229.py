# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        sin_x = x.sin()
        # element_size() returns the number of bytes per element (e.g., 4 for float32)
        # This is inlined as a constant in the graph after the PR fix
        return sin_x.element_size() + x.sin()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2)

