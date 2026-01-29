# torch.rand((), dtype=torch.cfloat)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return torch.sigmoid(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.cfloat)

