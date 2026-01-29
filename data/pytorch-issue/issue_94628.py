# Input shapes: A (10000, 10000), B (10000,), C (10000,) all with dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        A, B, C = inputs
        result1 = A @ B + A @ C  # Compute A@B + A@C
        result2 = A @ (B + C)    # Compute A@(B + C)
        return result1 - result2  # Return difference between the two results

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.randn(10000, 10000, dtype=torch.float32)
    B = torch.randn(10000, dtype=torch.float32)
    C = torch.randn(10000, dtype=torch.float32)
    return (A, B, C)

