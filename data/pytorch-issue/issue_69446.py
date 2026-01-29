# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Output shape after pooling

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Shape becomes [B, 16, 112, 112]
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random image tensor with shape (batch, channels, height, width)
    # Using batch size 1 for simplicity; can be scaled
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

