# torch.rand([], dtype=torch.float)  # inferred input shape (scalar)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.clone()  # Triggers the error when input is a problematic sparse scalar

def my_model_function():
    return MyModel()

def GetInput():
    indices = torch.zeros(0, 1, dtype=torch.long)  # Indices for sparse COO tensor
    values = torch.tensor([12.3])  # Single-element values tensor for scalar sparse
    size = torch.Size([])  # Empty size for scalar
    return torch.sparse_coo_tensor(indices, values, size)

