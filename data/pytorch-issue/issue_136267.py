# torch.rand(12, 34, 56, dtype=torch.float32, device="cuda")
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_dim = 1  # Matches example's first_along_dim=1
        self.second_dim = 2  # Matches example's then_along_dim=2

    def forward(self, t):
        a = t.abs().amax(dim=[self.first_dim], keepdim=True)
        b = t.abs().amax(dim=[self.first_dim, self.second_dim], keepdim=True)
        return a, b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(12, 34, 56, device="cuda", dtype=torch.float32)

