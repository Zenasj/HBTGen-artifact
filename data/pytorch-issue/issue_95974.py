# (torch.rand(1, 1), torch.rand(1))  # Input A (1x1 matrix) and B (1-element vector)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        A, B = inputs
        return torch.linalg.solve(A, B)

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.rand(1, 1, requires_grad=True)
    B = torch.rand(1, requires_grad=True)
    return (A, B)

