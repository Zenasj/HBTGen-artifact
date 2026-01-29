# torch.rand(2, 3).to_sparse()  # Inferred input shape and dtype (sparse COO tensor)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Uses zeros_like without memory format for sparse input (compliant with fixed behavior)
        return torch.zeros_like(x) + x  # Example operation using sparse tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Create a sparse COO tensor with shape (2,3)
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
    values = torch.tensor([3, 4, 5], dtype=torch.float32)
    sparse_size = (2, 3)
    return torch.sparse_coo_tensor(indices, values, sparse_size).coalesce()

