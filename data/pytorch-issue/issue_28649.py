# torch.rand(4, dtype=torch.float, device='cuda')  # Input is a 1D tensor of 4 elements on CUDA
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Indices from original issue's example, transposed and stored as a buffer on CUDA
        indices = torch.tensor(
            [
                [0, 1, 2, 3],  # Dimension 0 indices
                [2, 1, 1, 5],  # Dimension 1 indices
                [3, 2, 4, 1],  # Dimension 2 indices
            ],
            dtype=torch.int64,
            device='cuda'
        )
        self.register_buffer('indices', indices)

    def forward(self, values):
        # Create sparse COO tensor with fixed indices and input values
        sparse_tensor = torch.sparse_coo_tensor(
            indices=self.indices,
            values=values,
            size=(4, 6, 5),
            dtype=torch.float32,
            device='cuda'
        )
        # Sum over dimension 2 and convert to dense tensor
        summed = torch.sparse.sum(sparse_tensor, dim=2)
        dense = summed.to_dense()
        return dense  # Output shape: (4, 6)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching the values' shape and device
    return torch.rand(4, dtype=torch.float, device='cuda', requires_grad=True)

