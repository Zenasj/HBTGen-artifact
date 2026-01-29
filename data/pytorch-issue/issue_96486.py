# torch.rand(1, 1)  # Dummy input shape as the model does not use the input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        t1 = torch.rand(8, 8)
        t2 = torch.empty(8, 8)
        return t1 + t2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1)  # Dummy input to satisfy interface requirements

