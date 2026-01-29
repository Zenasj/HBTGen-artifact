# torch.rand(3, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        aa = torch.tensor([[0], [1], [2]])
        return aa.expand_as(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 2, dtype=torch.float32)

