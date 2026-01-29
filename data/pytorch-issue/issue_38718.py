# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # This triggers the JIT error when compiled with torch.compile
        return torch.nonzero(x, as_tuple=True)

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a 2D input tensor (shape 5x5) as a common test case
    return torch.rand(5, 5, dtype=torch.float32)

