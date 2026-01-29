# torch.rand(B, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def forward(self, x):
        if self.param.shape[0] < 10:
            return torch.mm(x, self.param)
        else:
            return torch.relu(torch.mm(x, self.param))

def my_model_function():
    return MyModel(in_channels=5)

def GetInput():
    return torch.randn(10, 5)

