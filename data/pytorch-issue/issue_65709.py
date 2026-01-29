# torch.rand(0, 2), torch.rand(0, 1)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        mean, std = inputs
        return torch.normal(mean=mean, std=std)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(0, 2), torch.rand(0, 1))

