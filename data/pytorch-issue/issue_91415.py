# torch.rand(B, M, M, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.linalg.lu_factor(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Batch dimension B=2, matrix size M=3 (square matrix requirement)
    return torch.rand(2, 3, 3, dtype=torch.float32)

