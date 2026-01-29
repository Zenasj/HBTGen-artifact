# torch.rand((), dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('foo', torch.zeros(()))

    def forward(self, x):
        self.foo = self.foo + 1
        return x + self.foo

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)

