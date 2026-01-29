# torch.rand(1, 1, 10, 1, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Trigger segfault by assigning scalar to .imag (per issue description)
        x.imag = 0  # This line will cause segmentation fault in vulnerable PyTorch versions
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 10, 1, dtype=torch.complex64)

