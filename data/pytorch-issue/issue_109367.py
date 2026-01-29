# torch.rand(3, 3)  # Input shape inferred for trace operation
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.trace(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3)

