# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, _):
        shape = (1, 3, 1000)
        return (torch.rand(shape, dtype=torch.double), torch.rand(shape, dtype=torch.float))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

