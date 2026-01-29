# torch.rand(2, 64, 64, dtype=torch.complex128)  # Inferred from linalg_svd test input (numel=4096 → 64x64 matrix)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute singular values using linalg.svd (one of the slow/failed operations in the issue's context)
        _, s, _ = torch.linalg.svd(x, full_matrices=False)
        return s

def my_model_function():
    # Returns a model instance representing an operation under test (e.g., linalg_svd)
    return MyModel()

def GetInput():
    # Generates a complex tensor matching the input shape of problematic tests (e.g., linalg_svd)
    return torch.rand(2, 64, 64, dtype=torch.complex128)

