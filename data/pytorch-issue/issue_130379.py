# torch.rand(4), torch.rand(4)  # Input is a tuple of two tensors of shape (4,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        with torch.no_grad():
            c = torch.tensor(4)
            z = c + x + y
        return z * z

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(4), torch.rand(4))

