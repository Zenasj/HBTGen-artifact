# torch.rand(1, 1, 2, 2, dtype=torch.half, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Workaround for argmax not supporting half in newer PyTorch versions
        return x.float().argmax()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random half-precision tensor on CUDA matching the required input shape
    return torch.rand(1, 1, 2, 2, dtype=torch.half, device='cuda')

