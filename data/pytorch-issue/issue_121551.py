# torch.rand(5), torch.rand(3, dtype=torch.float)  # Input is a tuple of two 1D tensors
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.einsum('i,j->ij', x, y)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(5), torch.rand(3))

