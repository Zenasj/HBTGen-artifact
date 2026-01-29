# (torch.rand(1), torch.rand(1)) ← inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1), torch.rand(1))

