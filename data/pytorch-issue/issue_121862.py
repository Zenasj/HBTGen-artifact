# torch.rand(1000, 3000, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1000, 3000))  # Matches original weight initialization

    def forward(self, x):
        return F.linear(x, self.w)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1000, 3000)  # Matches input shape from original code

