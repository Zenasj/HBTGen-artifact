# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        with torch.no_grad():
            out = x + 1
        out2 = x + 1  # Requires_grad=True because outside no_grad context
        return out, out2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, requires_grad=True)

