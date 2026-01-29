# torch.rand(1, 1000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param = nn.Parameter(torch.randn(1000, 1000))
    
    def forward(self, x):
        return x @ self.param  # Example forward pass using the parameter

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1000, dtype=torch.float32)

