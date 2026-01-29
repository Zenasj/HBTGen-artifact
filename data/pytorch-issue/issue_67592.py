# torch.rand(1000, dtype=torch.float32) ← Inferred input shape based on reduced size for feasibility
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('p', torch.tensor([0.2, 0.9], dtype=torch.float32))  # Quantile values from original example
    
    def forward(self, x):
        return torch.quantile(x, self.p)

def my_model_function():
    return MyModel()

def GetInput():
    # Reduced n from 2e7 to 1000 to avoid "input tensor is too large" error for demonstration
    n = 1000
    return torch.rand(n, dtype=torch.float32)

