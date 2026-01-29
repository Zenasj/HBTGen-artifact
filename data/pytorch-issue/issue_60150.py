# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        temp = torch.ones([4000, 4000], device='cuda')
        return temp  # Tensor causing memory allocation

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input to trigger forward pass

