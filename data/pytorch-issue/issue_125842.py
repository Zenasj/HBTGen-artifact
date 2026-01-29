# (torch.rand(3), torch.rand(3))  # Input is a tuple of two 1D tensors of shape (3,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, tensors):
        t1, t2 = tensors
        return torch.dot(t1, tensor=t2)

def my_model_function():
    return MyModel()

def GetInput():
    t1 = torch.rand(3)
    t2 = torch.rand(3)
    return (t1, t2)

