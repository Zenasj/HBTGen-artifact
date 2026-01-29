# torch.randint(0, 10, (8,), dtype=torch.int64)  # Inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x ** 2  # Reproduces the core operation causing Dynamo fallback

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (8,), dtype=torch.int64)  # Matches integer array input from original repro

