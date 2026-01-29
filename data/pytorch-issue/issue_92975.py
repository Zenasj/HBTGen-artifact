# torch.rand(B, dtype=torch.float, device='cuda')
import torch
import torch.nn.functional as F
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.randn(1, 3, device='cuda'))  # Matches y's shape from the example
        
    def forward(self, x):
        # Apply dropout to the fixed y parameter and compute mean
        dropped = F.dropout(self.y, p=0.3, training=self.training)
        meaned = dropped.mean(1)
        return x + meaned

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, device='cuda')  # Matches x's shape (3,) from the example

