# Input is a tuple of two 1D tensors of shape (0,): (torch.rand(0), torch.rand(0))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        t1, t2 = inputs
        return torch.matmul(t1, t2)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(0), torch.rand(0))

