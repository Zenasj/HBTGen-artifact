# Input shapes: (10000, 10000), (10000,), (10000,), dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        A, B, C = inputs
        result_1 = A @ B + A @ C
        result_2 = A @ (B + C)
        return (result_1 - result_2).norm()

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.randn(10000, 10000, dtype=torch.float32)
    B = torch.randn(10000, dtype=torch.float32)
    C = torch.randn(10000, dtype=torch.float32)
    return (A, B, C)

