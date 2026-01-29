# torch.rand(2, 2, dtype=torch.float64)  # Inferred input shape from problematic case
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x1):
        x0 = torch.ones_like(x1, dtype=torch.complex128)
        return x0 / x1

def my_model_function():
    return MyModel()

def GetInput():
    # Create tensor with shape (2,2) and inf at [0,0] to reproduce discrepancy
    x1 = torch.ones(2, 2, dtype=torch.float64)
    x1[0, 0] = torch.inf
    return x1

