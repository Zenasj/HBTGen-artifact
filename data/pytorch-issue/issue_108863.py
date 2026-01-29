# torch.rand(B, 64, C=16, H=32, W=32, dtype=torch.float32)  # Inferred from the original issue's input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 6)
        self.bn1 = nn.BatchNorm2d(16)
        
    def forward(self, x):
        return self.bn1(self.conv1(x))

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(64, 16, 32, 32, dtype=torch.float32)

