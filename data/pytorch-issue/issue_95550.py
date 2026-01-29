# torch.rand(2, 2, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Original path (problematic backward semantics)
        out1 = x.to_dense()
        # Corrected path (enforces non-masked semantics via sparse_mask)
        mask = x.detach()  # Preserve sparse structure for masking
        out2 = x.sparse_mask(mask).to_dense()
        # Return difference between outputs to indicate discrepancy
        return out1 - out2

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a sparse input tensor with random non-zero elements
    dense = torch.rand(2, 2, dtype=torch.float64)
    dense[dense < 0.5] = 0  # Introduce sparsity
    sparse_tensor = dense.to_sparse().requires_grad_()
    return sparse_tensor

