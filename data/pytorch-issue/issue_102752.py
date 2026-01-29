# torch.rand(16, 16, 16, dtype=torch.float32, device='cuda')  # Input shape and dtype inferred from the repro
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = x + 4
        z = torch.fft.fftn(y)
        return z

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(16, 16, 16, dtype=torch.float32, device='cuda')

