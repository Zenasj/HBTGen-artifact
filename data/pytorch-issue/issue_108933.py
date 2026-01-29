# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.color_mode = Color.RED  # Enum attribute causing TorchScript issue in Python 3.11
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)  # Dummy layers to process input

    def forward(self, x):
        x = self.conv(x)
        if self.color_mode == Color.RED:  # Enum comparison in forward pass
            x = x * 2
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()  # Returns model instance with problematic enum attribute

def GetInput():
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)  # Batched input matching model's expected shape

