# torch.rand(1, 2, 3, dtype=torch.float64, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        x.retain_grad()
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3, dtype=torch.float64, requires_grad=True)

