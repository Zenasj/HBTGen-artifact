# torch.rand(100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.as_strided(x, size=(2, 2), stride=(25, 40), storage_offset=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100)

