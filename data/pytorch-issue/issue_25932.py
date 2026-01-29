# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        with torch.no_grad():
            y = x.expand(1)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a scalar tensor with requires_grad=True as in the original example
    return torch.rand((), requires_grad=True, dtype=torch.float32)

