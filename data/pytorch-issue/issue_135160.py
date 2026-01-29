# torch.rand(16, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.view(torch.int32) >> 2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(16, 16, dtype=torch.float32)

