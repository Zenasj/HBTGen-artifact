# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.add_tensor = nn.Parameter(torch.randn(16, 1, 1, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        return x + self.add_tensor.expand_as(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

