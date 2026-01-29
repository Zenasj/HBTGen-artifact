# torch.rand(2, 3, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dims = [3, 2, 1, 0]  # Replicates original dims list from issue's code

    def forward(self, x):
        return torch.flip(x, self.dims)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, 32, 32, dtype=torch.float32)

