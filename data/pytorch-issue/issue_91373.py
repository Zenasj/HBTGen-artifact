# torch.rand(B, 3, dtype=torch.double)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters, just computes different norm implementations
        # Submodules not needed here as these are simple function calls

    def forward(self, x):
        # Compute all compared norm methods
        linalg_norm = torch.linalg.norm(x, dim=1)
        vector_norm = torch.linalg.vector_norm(x, dim=1)
        custom_norm = torch.sqrt((x * x).sum(dim=1))
        return linalg_norm, vector_norm, custom_norm

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape and dtype from benchmark (double precision issue)
    return torch.rand(200000, 3, dtype=torch.double)

