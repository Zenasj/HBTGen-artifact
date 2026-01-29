# torch.randint(-10, -9, (1, 2), dtype=torch.int64), torch.randn((2, 32), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        x = x + 10
        y[x] += y[x]  # In-place update using indices
        return y

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randint(-10, -9, (1, 2), dtype=torch.int64)
    y = torch.randn((2, 32), dtype=torch.float32)
    return (x, y)  # Tuple of input tensors (indices and values)

