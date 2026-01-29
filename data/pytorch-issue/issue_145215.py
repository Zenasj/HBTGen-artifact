# torch.rand(1, 1, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Converts sparse input to dense tensor, which triggers OpenMP threading issue
        return x.to_dense()

def my_model_function():
    # Returns the model instance that demonstrates the sparse-to-dense conversion
    return MyModel()

def GetInput():
    # Creates a 4D sparse tensor matching the expected input shape (B=1, C=1, H=2, W=2)
    indices = torch.tensor([
        [0, 0],  # Batch indices
        [0, 0],  # Channel indices
        [0, 1],  # Height indices (original row indices)
        [1, 0]   # Width indices (original column indices)
    ], dtype=torch.int64)
    values = torch.tensor([1, 1], dtype=torch.float32)
    size = torch.Size([1, 1, 2, 2])  # (Batch, Channel, Height, Width)
    return torch.sparse_coo_tensor(indices, values, size)

