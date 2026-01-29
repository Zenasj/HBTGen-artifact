# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape (Batch, Channels, Height, Width)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

