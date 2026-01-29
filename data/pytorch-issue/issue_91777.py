# torch.rand(1024, 1024, dtype=torch.float32), torch.rand(1024, 1024, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x @ y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(1024, 1024, dtype=torch.float32), torch.randn(1024, 1024, dtype=torch.float32))

