# torch.rand(B, 3, 128, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Basic CNN structure matching input shape (3 channels, 128x128 images)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Returns initialized model instance
    return MyModel()

def GetInput():
    # Returns random tensor matching input requirements
    return torch.rand(32, 3, 128, 128, dtype=torch.float32)

