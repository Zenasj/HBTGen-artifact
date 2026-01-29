# torch.rand(6, dtype=torch.float32)
import torch
from collections import namedtuple

MyTuple = namedtuple('MyTuple', ['a', 'b', 'c'])

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Split input tensor into three parts of equal size
        a, b, c = torch.split(x, 2)
        return MyTuple(a, b, c)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(6)

