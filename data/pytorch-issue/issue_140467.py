# torch.rand(64, 1024, 8, 64, dtype=torch.half, device='cuda')  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.nn.functional.scaled_dot_product_attention(x, x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(64, 1024, 8, 64, dtype=torch.half, device='cuda')

