# torch.rand(60, dtype=torch.float32)
import torch
from torch import nn

def nemo(x):
    return x + 1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(torch.jit.is_tracing())
        if torch.jit.is_tracing():
            return x
        else:
            return nemo(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(60, dtype=torch.float32)

