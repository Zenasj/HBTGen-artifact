# torch.rand(N, dtype=torch.float16, device='cpu')  # Input is a 1D tensor of shape (N,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        N = x.size(0)
        A = torch.rand(N, N, dtype=x.dtype, device=x.device)
        return torch.matmul(A, x)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the problematic case with N=50000 as in the original issue
    return torch.rand(50000, dtype=torch.float16, device='cpu')

