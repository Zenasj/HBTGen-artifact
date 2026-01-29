# torch.rand(1024, dtype=torch.float32, device="cuda")  # Inferred input shape from the reproducer example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        idx = torch.arange(a.size(0), device=a.device)
        padded_idx = torch.constant_pad_nd(idx, (1050, 0))
        padded_idx = torch.where(padded_idx >= 0, padded_idx, padded_idx)  # Replicates the redundant where() call from the issue
        return a[padded_idx]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1024, dtype=torch.float32, device="cuda")

