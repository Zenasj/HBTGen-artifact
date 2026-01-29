# torch.randn(1, 3, 224, 224, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class NewConv2d(nn.Conv2d):
    def forward(self, x):
        return super().forward(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = NewConv2d(3, 16, kernel_size=1, stride=2)  # Matches the original model structure
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32, device='cuda')  # Matches the input shape and device

