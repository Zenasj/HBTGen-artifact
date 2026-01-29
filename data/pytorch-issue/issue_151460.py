# torch.randint(0, 100, (32,), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply bitwise_not_() on a strided view (every 2nd element)
        strided_view = x[::2]
        strided_view.bitwise_not_()
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 1D tensor matching the example's input shape and dtype
    return torch.randint(0, 100, (32,), dtype=torch.int32)

