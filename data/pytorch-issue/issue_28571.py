# torch.rand(2, 2, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return ~x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.BoolTensor(2, 2)

