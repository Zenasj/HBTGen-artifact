# torch.rand(1, dtype=torch.float32)  # Dummy input, not used
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Include both COO and CSR buffers to demonstrate comparison
        self.register_buffer("a_coo", torch.eye(3).to_sparse())  # Working COO format
        self.register_buffer("a_csr", torch.eye(3).to_sparse_csr())  # Failing CSR format

    def forward(self, x):
        # Dummy forward to accept input (required for GetInput compatibility)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input compatible with forward()

