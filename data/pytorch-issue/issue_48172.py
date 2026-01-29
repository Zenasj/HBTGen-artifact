# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for a standard image model
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 224 * 224, 10)  # Example layer to trigger memory allocation

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a simple model instance with basic layers
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the assumed shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

