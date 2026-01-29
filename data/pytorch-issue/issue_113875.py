# torch.rand(5, 1, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        if x.size() != (5, 1, 2, 3):
            return x.cos()
        else:
            return x.sin()

def my_model_function():
    return MyModel()

def GetInput():
    H = torch.randint(1, 6, (1,)).item()  # Random H between 1-5
    W = torch.randint(1, 6, (1,)).item()  # Random W between 1-5
    return torch.rand(5, 1, H, W, dtype=torch.float32)

