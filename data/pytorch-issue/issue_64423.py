# torch.rand(B, 32, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)

def my_model_function():
    # Return an instance of MyModel, initialized with dim=64 as in the original issue
    return MyModel(64)

def GetInput():
    # Return input matching the Affine model's expected dimensions (batch, features, dim=64)
    return torch.randn(1, 32, 64)

