# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    __constants__ = ['linears']

    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10)

