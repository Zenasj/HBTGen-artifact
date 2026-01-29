# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 10)

