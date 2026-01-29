# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(3, 16, 3))
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 32, 32, dtype=torch.float32)

