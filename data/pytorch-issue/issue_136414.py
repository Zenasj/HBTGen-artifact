# torch.rand(3, 4, dtype=torch.float32)  # Inferred input shape from logs
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.to(device="cuda", non_blocking=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4, dtype=torch.float32)

