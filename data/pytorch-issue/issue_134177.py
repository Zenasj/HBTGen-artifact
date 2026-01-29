# torch.rand(B, 4096, 40, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.einsum('b i d, b j d -> b i j', x, x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 16  # Base case from original code (n_samples=1)
    return torch.rand(B, 4096, 40, dtype=torch.float32)

