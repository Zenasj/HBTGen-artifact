# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed shape (1, 2, 32, 32) based on input structure
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.digamma()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 32, 32, dtype=torch.float32)

