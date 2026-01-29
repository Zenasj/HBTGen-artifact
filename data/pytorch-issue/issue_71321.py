# torch.rand(B, H, W, C, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute FFT on original tensor (dim (1,2))
        x1 = torch.fft.ifft2(x, norm="ortho", dim=(1, 2))
        
        # Permute to (B, C, H, W), make contiguous, then compute FFT on last two dims
        permuted = x.permute(0, 3, 1, 2).contiguous()
        x2_permuted = torch.fft.ifft2(permuted, norm="ortho", dim=(-2, -1))
        # Permute back to original shape (B, H, W, C)
        x2 = x2_permuted.permute(0, 2, 3, 1)
        
        # Calculate absolute error sum and check if exceeds threshold
        error = torch.abs(x1 - x2).sum()
        return error > 1e-6  # Boolean output indicating non-negligible error

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B, H, W, C) as per original example
    return torch.randn((1, 100, 50, 8), dtype=torch.complex64)

