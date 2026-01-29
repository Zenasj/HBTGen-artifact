import torch
from torch import nn

# torch.rand(1, 3, 33, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims  # Permutation dimensions stored as model parameter

    def forward(self, x):
        return x.permute(self.dims)

def my_model_function():
    # Returns model with permutation dimensions (0, 2, 1) as in original issue
    return MyModel(dims=(0, 2, 1))

def GetInput():
    # Generates 3D tensor matching the original input shape (1, 3, 33)
    return torch.rand(1, 3, 33, dtype=torch.float32)

