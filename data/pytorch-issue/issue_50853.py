# torch.rand(2**18, dtype=torch.float32)  # Inferred input shape (1D tensor of size 2^18)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x + x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2**18, dtype=torch.float32)

