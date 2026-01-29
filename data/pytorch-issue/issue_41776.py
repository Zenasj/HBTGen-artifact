# torch.rand(14, 7, dtype=torch.float32)  # Valid input shape from geqrf output
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        qr, tau = torch.geqrf(x)
        return torch.orgqr(qr, tau)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(14, 7, dtype=torch.float32)

