# torch.rand(1, 2, 2, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.tensor(1e-3, device='cpu'))  # Scalar on CPU as per the PR fix

    def forward(self, x):
        return torch.addcdiv(x, x, self.b)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 2, 1, device='cuda', dtype=torch.float32)

