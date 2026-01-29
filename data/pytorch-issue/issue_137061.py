# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sum(x ** 2)

def my_model_function():
    return MyModel()

def GetInput():
    # Using default shape from the first example (batch_size=5, vector_length=4)
    return torch.randn(5, 4)

