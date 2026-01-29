import torch
from torch.nn import Linear, Module

# torch.rand(4, 512, 1024, dtype=torch.float32)  # Input shape (B, S, F)
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(1024, 4096)
        self.linear2 = Linear(4096, 1024)

    def forward(self, x):
        y = self.linear(x)
        z = self.linear2(y)
        return (z,)  # Returns a tuple as in the original example

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 512, 1024, dtype=torch.float32)

