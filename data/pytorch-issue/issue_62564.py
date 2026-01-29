# torch.rand(1, dtype=torch.float32)  # Input shape: (1,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x * x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, requires_grad=True)

