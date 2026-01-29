# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Create tensors using all affected new_* methods with requires_grad=True
        ones = x.new_ones((1,), requires_grad=True)
        zeros = x.new_zeros((1,), requires_grad=True)
        empty = x.new_empty((1,), requires_grad=True)
        full = x.new_full((1,), 5.0, requires_grad=True)
        return ones, zeros, empty, full

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

