# torch.rand(1, 1, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)  # Matches input channels=1 and kernel size 3

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 32, 32, dtype=torch.float32)

