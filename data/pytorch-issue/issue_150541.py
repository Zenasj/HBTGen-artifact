# torch.rand(1024, 512, dtype=torch.float), torch.rand(512, 1, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        x, y = inputs
        return torch.mm(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1024, 512)
    y = torch.rand(512, 1)
    return (x, y)

