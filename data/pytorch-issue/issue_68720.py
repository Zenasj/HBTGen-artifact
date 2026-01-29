# torch.rand(2, 3, 3, dtype=torch.double)  # Symmetric matrices of batch size 2, 3x3 dimensions
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Cholesky decomposition check (returns True if SPD)
        _, info = torch.linalg.cholesky_ex(x)
        cholesky_result = info.eq(0)  # Shape: (batch_size, )

        # Eigenvalue check (returns True if all eigenvalues are positive)
        eigvals = torch.linalg.eigvalsh(x)  # For symmetric matrices
        eigen_result = (eigvals > 0).all(dim=-1)  # Shape: (batch_size, )

        # Return True where the two methods disagree
        return torch.logical_xor(cholesky_result, eigen_result)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    N = 3  # Matrix dimension
    A = torch.rand(B, N, N, dtype=torch.double)
    # Ensure symmetry by averaging with transpose
    sym_A = (A + A.transpose(1, 2)) / 2
    return sym_A

