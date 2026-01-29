# torch.rand(B, C, L, dtype=torch.float32)  # Input shape (Batch, Channels, Length)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.fft.fft(x, n=2, dim=-1, norm=None)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, dtype=torch.float32, device='cuda')

