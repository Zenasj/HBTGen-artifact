# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.to("cuda:0")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1)

