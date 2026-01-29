# torch.rand(1, 1, 1, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2  # Reproduces the computation leading to the Node binding issue

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching the input shape expected by MyModel
    # with requires_grad to trigger autograd graph construction
    return torch.rand(1, 1, 1, 10, requires_grad=True)

