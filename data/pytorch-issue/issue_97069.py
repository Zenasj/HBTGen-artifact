# torch.rand(1073741824, dtype=torch.float16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return x  # Pass-through to replicate memory inspection scenario

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024*1024*1024, dtype=torch.float16, device='cuda')

