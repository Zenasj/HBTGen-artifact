# torch.rand(1, 2, 3, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand(2, 3, dtype=torch.float32, device='cuda', requires_grad=True))

    def forward(self, x):
        return x  # Dummy forward to accept input

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3, dtype=torch.float32, device='cuda')

