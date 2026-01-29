# torch.rand((), dtype=torch.half)  # 0-dimensional tensor with half precision
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert dense tensor to sparse, then permute with empty dimensions (triggering the segfault)
        sparse_x = x.to_sparse()
        return sparse_x.permute(())  # Empty tuple for 0-dimensional permutation

def my_model_function():
    return MyModel()

def GetInput():
    # Create 0-dimensional tensor with half-precision (matches the issue's test case)
    return torch.randn((), dtype=torch.half)

