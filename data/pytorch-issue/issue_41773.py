# torch.rand(20, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(20, 5, dtype=torch.float32)

