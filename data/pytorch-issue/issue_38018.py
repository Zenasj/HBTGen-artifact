# torch.rand(2, 3, dtype=torch.float64, device='cuda')
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute CUDA-based median (buggy path)
        cuda_med = torch.median(x, dim=1).values
        # Compute CPU-based median (correct reference)
        cpu_med = torch.median(x.cpu(), dim=1).values
        # Compare with NaN-handling and return boolean result
        equal = torch.allclose(cuda_med.cpu(), cpu_med, atol=1e-5, equal_nan=True)
        return torch.tensor([equal], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create tensor with NaNs except first column (1.1) on CUDA
    x = torch.full((2, 3), np.nan, dtype=torch.float64, device='cuda')
    x[:, :1] = 1.1
    return x

