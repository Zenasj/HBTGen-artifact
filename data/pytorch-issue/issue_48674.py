# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x[-3]  # Fails when scripted due to negative index AST differences

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32)

