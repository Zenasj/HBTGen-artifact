# torch.rand(2), torch.rand(2)  # Input is a tuple of two tensors of shape (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        a, b = x
        c = torch.cat((a, b))
        return c.min(0).values

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(2)
    b = torch.rand(2)
    return (a, b)

