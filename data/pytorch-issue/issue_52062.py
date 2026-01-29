# torch.rand(8, dtype=torch.bool, device='cuda')  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Slice every second element (non-contiguous view)
        sliced = x[::2]
        # Apply flip along the first dimension (triggers the error on older PyTorch versions)
        return torch.flip(sliced, [0])

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a boolean tensor of shape (8,) on CUDA
    return torch.rand(8, device='cuda') > 0.5

