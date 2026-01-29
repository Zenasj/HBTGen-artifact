# torch.rand(1, 3, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = dict([('a', x)])  # Reproduces the Dynamo-unsupported dict(list) pattern
        return y['a']

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 1, 1, dtype=torch.float32)

