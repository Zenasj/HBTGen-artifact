# (torch.rand(2, 3, 3, dtype=torch.float32), torch.rand(2, 3, 4096, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        B, A = inputs  # B: (2,3,3) coefficient matrix, A: (2,3,4096) RHS
        return torch.linalg.solve(B, A)  # Solves B X = A

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tuple of (B, A) tensors on CUDA device
    B = torch.randn(2, 3, 3, dtype=torch.float32).cuda()
    A = torch.randn(2, 3, 4096, dtype=torch.float32).cuda()
    return (B, A)

