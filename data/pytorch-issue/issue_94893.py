# torch.rand(B, 768, 1, D, dtype=torch.float32)  # B=1, D varies between 8000-15000
import torch
import random
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Apply torch.fft.rfft along last dimension (D) with n=x.size(-1)
        return torch.fft.rfft(x, n=x.size(-1), dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input with varying last dimension (D)
    B = 1
    C = 768
    H = 1
    D = random.randint(8000, 15000)
    return torch.randn(B, C, H, D, dtype=torch.float32)

