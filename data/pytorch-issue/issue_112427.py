# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.LogSigmoid()
    def forward(self, x):
        return self.m(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2)

