# torch.rand(B, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.diff(x, dim=-1).sum(-1)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 2D tensor with shape (batch_size, 4) as per error context
    return torch.rand(2, 4, dtype=torch.float32)

