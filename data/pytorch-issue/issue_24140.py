# torch.rand(2, 3, dtype=torch.float)  # Input is converted to sparse COO format
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.sparse.sum(x)

def my_model_function():
    return MyModel()

def GetInput():
    dense = torch.randn(2, 3)
    return dense.to_sparse()  # Returns a sparse COO tensor as input

