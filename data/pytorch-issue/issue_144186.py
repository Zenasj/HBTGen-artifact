# torch.rand(8, 8, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.sum(dim=-1)
        x = self.softmax(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(8, 8, 2, dtype=torch.float32)

