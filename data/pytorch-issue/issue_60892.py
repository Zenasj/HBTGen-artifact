# torch.rand(B, 1, 20, 20, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        matrix = x.squeeze(0).squeeze(0)  # Extract the 20x20 matrix from (B,1,20,20)
        return torch.linalg.eigh(matrix)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1
    N = 20
    a = torch.randn(B, 1, N, 50, dtype=torch.float32)
    # Create symmetric matrix via A @ A^T + diagonal
    matrix = a @ a.transpose(-2, -1) + 1e-3 * torch.eye(N).view(1, 1, N, N)
    return matrix.to(torch.float32)

