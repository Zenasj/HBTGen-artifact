# torch.rand(64, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, a):
        na = torch.linalg.vector_norm(a, dim=self.dim)
        eps = 1e-8
        return na.clamp_min(eps)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(64, dtype=torch.float16)

