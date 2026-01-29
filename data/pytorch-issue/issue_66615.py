# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape for a typical image model (batch=1, 3 channels, 224x224)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Returns a simple CNN model instance
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

