# torch.rand(4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nonzero(x > 0)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.ones(4, 4)
    x[0, 3] = 0
    x[1, 2] = 0
    return x

