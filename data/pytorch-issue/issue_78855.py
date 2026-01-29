# torch.rand(2, 63, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x[0], x[1]
        return torch.fmod(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    N = 63  # Maximum n from the original test loop
    a = torch.randn(N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)
    a[0] = 8.0  # Fixed index 0 for reproducibility
    b[0] = 2.0e-38
    return torch.stack([a, b])  # Returns a tensor of shape (2, 63)

