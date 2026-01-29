# torch.rand(1, 1, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 50, 2)

    def forward(self, x):
        out = self.conv(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 224, 224, dtype=torch.float32)

