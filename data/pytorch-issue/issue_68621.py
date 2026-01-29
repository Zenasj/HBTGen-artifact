# torch.rand(1, 3, 3, 1, dtype=torch.float32, device="cuda", requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize a sparse CSR matrix as a buffer
        dense = torch.randn(3, 3, device="cuda", requires_grad=True)
        self.register_buffer("M", dense.to_sparse_csr())

    def forward(self, x):
        # Reshape input to 2D for CSR matrix multiplication
        x_reshaped = x.view(3, 3)
        y = torch.sparse.mm(self.M, x_reshaped)  # Explicitly use sparse mm
        # Reshape output back to 4D for consistency with input shape
        return y.view(1, 3, 3, 1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 3, 1, dtype=torch.float32, device="cuda", requires_grad=True)

