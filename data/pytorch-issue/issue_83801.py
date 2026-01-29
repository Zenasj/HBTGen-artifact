# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.cat((x, x), dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, dtype=torch.float32)

