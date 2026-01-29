# torch.rand(1, 1, 1028, 1028, dtype=torch.float32)
import torch
from torch import nn

class OldArgMin(nn.Module):
    def forward(self, x, dim):
        reversed_x = torch.flip(x, (dim,))
        reversed_indices = reversed_x.argmin(dim=dim)
        original_indices = x.size(dim) - 1 - reversed_indices
        return original_indices

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old = OldArgMin()
        self.dim = 1  # Fixed dimension as per example

    def forward(self, x):
        new_res = torch.argmin(x, dim=self.dim)
        old_res = self.old(x, self.dim)
        return (new_res == old_res).all().float()

def my_model_function():
    return MyModel()

def GetInput():
    # Create a 2D tensor with shape (1028, 1028) where some rows have duplicate minima
    x = torch.rand(1028, 1028)
    x[:, 0] = 0.0  # Ensure duplicates in min values along dim=1
    x[:, -1] = 0.0
    return x

