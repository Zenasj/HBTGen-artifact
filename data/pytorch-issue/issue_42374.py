# torch.rand(1, 2, dtype=torch.int) and torch.rand(1, 2, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return a.add(b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randint(0, 2, (1, 2), dtype=torch.int)  # Matches original example's shape and dtype
    b = torch.rand(1, 2, dtype=torch.bfloat16)
    return (a, b)

