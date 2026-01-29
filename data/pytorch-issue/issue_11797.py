# torch.rand(B, 250, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, N=250):
        super().__init__()
        self.A = nn.Parameter(torch.diag(torch.full((N,), 2.0, dtype=torch.float64)))
    
    def forward(self, x):
        return torch.matmul(x, self.A)

def my_model_function():
    return MyModel()

def GetInput():
    N = 250
    return torch.rand(1, N, dtype=torch.float64, requires_grad=True)

