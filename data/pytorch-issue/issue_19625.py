# torch.rand(10, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create pinned CPU memory and perform copy with non-blocking flag
        y = torch.empty(x.size(), pin_memory=True)
        y.copy_(x, non_blocking=True)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Generate 1D tensor on CUDA as per the original bug example
    return torch.rand(10, dtype=torch.float32, device='cuda')

