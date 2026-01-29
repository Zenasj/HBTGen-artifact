# torch.rand(3, 4, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # PyTorch implementation (buggy behavior)
        pt_sum = torch.sum(x, dim=())
        # Numpy-like implementation (expected behavior: no reduction)
        np_sum = x  # shape remains unchanged
        # Compare if PyTorch's output matches numpy-like output's shape
        shape_match = torch.tensor(pt_sum.shape == x.shape, dtype=torch.bool)
        return shape_match

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, 5)

