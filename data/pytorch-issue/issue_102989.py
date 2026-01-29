# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        p = torch.nn.Parameter(x)
        return p[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)

