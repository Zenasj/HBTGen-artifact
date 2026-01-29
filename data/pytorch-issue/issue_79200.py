# torch.rand(2, 4, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.abs(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (2, 4), dtype=torch.int64)

