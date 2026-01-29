# torch.rand(10, dtype=torch.bool, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return torch.roll(x, 1, 0)  # Reproduce the problematic operation

def my_model_function():
    return MyModel()

def GetInput():
    # Return CUDA bool tensor as in the original example
    return torch.zeros(10, dtype=torch.bool, device='cuda')

