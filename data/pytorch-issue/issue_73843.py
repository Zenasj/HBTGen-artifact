# torch.rand(1, 2, 9, 9, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 1, 1, 1))
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 9, 9, dtype=torch.float32)

