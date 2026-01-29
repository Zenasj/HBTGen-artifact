# torch.rand(2, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize CSR tensor as a buffer with requires_grad
        crow_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        col_indices = torch.tensor([0, 1], dtype=torch.int64)
        values = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        csr_tensor = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, dtype=torch.float32
        )
        self.register_buffer('csr_tensor', csr_tensor)
        self.csr_tensor.requires_grad_(True)  # Enable gradients

    def forward(self, x):
        # Problematic conversion that triggers the autograd error
        csr2 = self.csr_tensor.to_sparse(layout=torch.sparse_csr)
        y = torch.matmul(csr2, x)
        return y.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor matching the input expected by MyModel's forward (shape 2x1)
    return torch.ones((2, 1), dtype=torch.float32)

