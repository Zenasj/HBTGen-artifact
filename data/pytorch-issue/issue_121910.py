# torch.rand(B, 2), torch.rand(B + 1, 2)  # input tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # Unpack the tuple input
        return x + y[1:]  # Slice y's first dimension and add to x

def my_model_function():
    return MyModel()

def GetInput():
    B = torch.randint(1, 4, (1,)).item()  # Random B in [1, 2, 3]
    x = torch.rand(B, 2)                 # First input tensor (B, 2)
    y = torch.rand(B + 1, 2)             # Second input tensor (B+1, 2)
    return (x, y)                        # Return as a tuple

