# torch.rand(1), torch.rand(1)  # two tensors of shape (1,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        x += y  # In-place addition causing discrepancy in Dynamo/NNC
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1), torch.rand(1))

