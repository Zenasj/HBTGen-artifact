# torch.rand(1, 4, 1, 1, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return x * x + x - 3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 1, 1, dtype=torch.float32, device='cuda')

