# torch.rand(3, 3, dtype=torch.float32)  # Input is a sparse COO tensor of shape (3, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # CSR caching-optimized model structure (sparse matrix multiplication)
        self.dense_weight = nn.Parameter(torch.randn(3, 3))  # Dense weight for CSR-based matmul

    def forward(self, x):
        # Ensure CSR caching is triggered via coalesce()
        x = x.coalesce()
        # Perform CSR-optimized sparse-dense matrix multiplication
        return torch.sparse.mm(x, self.dense_weight)

def my_model_function():
    return MyModel()

def GetInput():
    # Create sparse COO tensor as per issue example
    indices = torch.LongTensor([[0, 1, 2], [2, 1, 0]])
    values = torch.FloatTensor([1, 1, 1])
    size = torch.Size([3, 3])
    return torch.sparse_coo_tensor(indices, values, size)

