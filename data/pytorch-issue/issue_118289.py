# torch.rand(B, 3, 256, 256, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        # Constant is a learnable parameter to avoid dynamic shape constraints on inputs
        self.constant = nn.Parameter(torch.ones(1, 32, 256, 256))  # Broadcastable across batch

    def forward(self, x):
        a = self.conv(x)
        a.add_(self.constant)  # Broadcast constant across batch dimension
        return self.maxpool(self.relu(a))

def my_model_function():
    return MyModel()

def GetInput():
    # Example batch size (can be any size due to parameter broadcasting)
    B = 2
    return torch.randn(B, 3, 256, 256, dtype=torch.float32)

