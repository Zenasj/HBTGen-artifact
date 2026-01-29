# torch.rand(1, 1, 100, 100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.sqrt(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 100, 100, dtype=torch.float32)

