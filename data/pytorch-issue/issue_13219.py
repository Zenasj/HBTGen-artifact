# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)  # Matches input channel (3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 100)  # 32 channels, 8x8 from pooling
        self.fc2 = nn.Linear(100, 10)  # Output 10 classes as in the issue's check

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    return torch.rand(64, 3, 32, 32, dtype=torch.float32)  # Matches input shape [64,3,32,32]

