# torch.rand(2, requires_grad=True)  # Inferred input shape from the example
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class GModule(nn.Module):
    def forward(self, x):
        return x.sin().sin()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_part = GModule()

    def forward(self, x):
        x = x * x
        x = x + 1
        return checkpoint(self.g_part, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, requires_grad=True)

