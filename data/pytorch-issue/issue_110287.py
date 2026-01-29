# torch.rand(2, 3, dtype=torch.float32)
import torch
import itertools
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the itertools.accumulate scenario that caused Dynamo error
        accumulated = itertools.accumulate([x, x])  # Triggers sum() on tensors
        return x * 2  # The actual output, but accumulation is part of the computation path

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

