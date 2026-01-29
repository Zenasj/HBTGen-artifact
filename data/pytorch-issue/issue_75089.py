# torch.rand(4, dtype=torch.bfloat16, device='cuda')  # Inferred input shape from the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.clamp(min=1.0) * 0.1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, device="cuda", dtype=torch.bfloat16)

