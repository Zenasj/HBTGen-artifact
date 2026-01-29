# torch.rand(B, 4, 4, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        eigenvalues, eigenvectors = torch.linalg.eigh(x)
        return eigenvectors  # Output eigenvectors to trigger backward issues on non-unique eigenvalues

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Matches input shape from issue example
    # Generate symmetric random matrix to ensure torch.linalg.eigh validity
    x = torch.rand(B, 4, 4, dtype=torch.float64)
    x = x + x.transpose(-2, -1)  # Enforce symmetry
    return x

