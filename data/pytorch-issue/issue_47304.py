# torch.rand(1, dtype=torch.double)  # Input is a 1-element tensor of doubles on CUDA
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.exp(x**2)

def my_model_function():
    # Returns the model in double precision on CUDA
    return MyModel().double().to('cuda')

def GetInput():
    # Returns a random tensor matching the input shape and requirements
    return torch.rand(1, dtype=torch.double, requires_grad=True, device='cuda')

