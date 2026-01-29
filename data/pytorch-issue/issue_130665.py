# torch.rand((), dtype=torch.float32)  # Scalar input
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x=None):
        if x is None:
            return torch.atleast_1d()  # Triggers the bug when called without input
        else:
            return torch.atleast_1d(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(())

