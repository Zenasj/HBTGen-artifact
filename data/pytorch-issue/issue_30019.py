# torch.rand(1, 3, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    __constants__ = ['my_const']
    def __init__(self):
        super(MyModel, self).__init__()
        self.my_const = 2  # Reproduces constant inlining issue when scripted

    def forward(self, x):
        return x + self.my_const

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

