# torch.rand(B, C, H, W, dtype=torch.float32)  # e.g., (1, 3, 100, 100)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, op=None):
        super().__init__()
        self.op = op  # Can be None or a torch.nn.Module instance
    
    def forward(self, x):
        if self.op is not None:
            x = self.op(x)
        return x

def my_model_function():
    # Returns a model instance with op=None (no operation)
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected shape
    return torch.rand(1, 3, 100, 100, dtype=torch.float32)

