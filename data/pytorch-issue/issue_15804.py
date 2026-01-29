# torch.randint(low=0, high=100000, size=(10000000,))  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare non-dim and dim implementations of torch.unique with return_inverse=True
        # Both should produce the same inverse indices for 1D tensors
        _, inv_non_dim = torch.unique(x, sorted=False, return_inverse=True)
        _, inv_dim = torch.unique(x, sorted=False, return_inverse=True, dim=0)
        return torch.all(inv_non_dim == inv_dim)  # Return comparison result

def my_model_function():
    # Returns a model instance that compares the two torch.unique implementations
    return MyModel()

def GetInput():
    # Returns a 1D tensor matching the input shape expected by MyModel
    return torch.randint(low=0, high=100000, size=(10000000,), dtype=torch.int64)

