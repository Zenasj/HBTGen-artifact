# torch.rand(1, 5, dtype=torch.float32)  # Inferred input shape based on issue example
import torch
import torch.nn as nn

@torch.fx.wrap
def foo(x):
    if x.ndim > 1:
        return torch.relu(x)
    else:
        return torch.neg(x)

class MyModel(nn.Module):
    def forward(self, x):
        return foo(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 5, dtype=torch.float32)

