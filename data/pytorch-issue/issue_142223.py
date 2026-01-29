# torch.rand(64, 64, dtype=torch.float32)  # Inferred input shape and dtype
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.tanh(x)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (64, 64) and dtype float32
    return torch.randn(64, 64, dtype=torch.float32)

