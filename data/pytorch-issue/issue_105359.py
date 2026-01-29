# torch.rand(1, 1, 50, 50, dtype=torch.float32, device='cuda')  # Input shape for MyModel
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x.squeeze()  # Convert to 2D tensor (50x50)
        # Compute in float32
        x_f32 = x.to(dtype=torch.float32)
        eigvals_f32, _ = torch.linalg.eigh(x_f32)
        
        # Compute in float64 (workaround)
        x_f64 = x.to(dtype=torch.float64)
        eigvals_f64, _ = torch.linalg.eigh(x_f64)
        
        return eigvals_f32, eigvals_f64  # Return both for comparison

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 1, 50, 50
    rank = 7  # Low-rank approximation to match issue's context
    a = torch.rand(B, C, H, rank, dtype=torch.float32, device='cuda')
    b = torch.rand(B, C, W, rank, dtype=torch.float32, device='cuda')
    matrix = a @ b.transpose(-2, -1)  # Rank-7 matrix
    matrix = matrix + matrix.transpose(-2, -1)  # Ensure symmetry
    return matrix

