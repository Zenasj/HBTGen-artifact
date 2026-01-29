# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        def outer(a):
            return a + 1
        def inner(b):
            return b + 2
        def deep(c):
            d = outer(c)  # Accesses outer from closure
            return inner(d)
        return deep(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

