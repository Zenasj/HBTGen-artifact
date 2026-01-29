# torch.rand(10**6, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.prod(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(10**6, dtype=torch.float32, device='cuda')

