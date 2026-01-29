# torch.rand(4, 4, dtype=torch.float)  # Input shape inferred from the issue's example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute both sum operations to compare dim=0 vs dim=-2 (the issue's core comparison)
        sum_dim0 = torch.sparse.sum(x, dim=0)
        sum_dim_neg2 = torch.sparse.sum(x, dim=-2)
        return (sum_dim0, sum_dim_neg2)  # Return both results for comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Create a sparse tensor matching the issue's example
    dense = torch.tensor([[1., 0., 0., 1.],
                         [0., 1., 0., 0.],
                         [0., 1., 1., 0.],
                         [0., 1., 0., 2.]])
    return dense.to_sparse()

