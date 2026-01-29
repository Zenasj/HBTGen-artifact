# Input is a tuple of (torch.rand(1, 10, 5), torch.rand(1, 10, 5, 10, 5))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x1, x2 = inputs
        return torch.cdist(x1, x2)

def my_model_function():
    return MyModel()

def GetInput():
    B, P, R, M = 1, 10, 10, 5
    x1 = torch.randn(B, P, M, dtype=torch.float32)
    x2 = torch.randn(B, R, M, R, M, dtype=torch.float32)
    return (x1, x2)

