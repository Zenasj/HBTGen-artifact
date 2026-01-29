# torch.rand(1, 1, 10, 10, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        zero = torch.zeros_like(x)
        return x.fmod(zero)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 4D tensor with shape (1,1,10,10) matching the input expectation
    return torch.randint(-9, 9, (1, 1, 10, 10), dtype=torch.int64)

