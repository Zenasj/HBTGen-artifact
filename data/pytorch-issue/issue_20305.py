# torch.rand(10, dtype=torch.float32, device='cuda')  # Inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, y):
        x = torch.empty_like(y, pin_memory=True, device='cpu')
        x.copy_(y, non_blocking=True)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, device='cuda')

