# torch.rand(7), torch.rand(8), torch.rand(6)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, z = inputs
        if (
            2 * x.shape[0] == y.shape[0] + z.shape[0]
            and x.shape[0] == z.shape[0] + 1
            and 2 * z.shape[0] == y.shape[0] + 4
        ):
            return 2 * y
        else:
            return 2 + y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(7), torch.rand(8), torch.rand(6))

