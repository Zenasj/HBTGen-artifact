# torch.rand(B, 1, 28, 28, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    mean = 0.1307  # MNIST mean
    std = 0.3081   # MNIST std
    img = torch.rand(1, 28, 28)  # Generate random image (C=1, H=28, W=28)
    normalized = (img - mean) / std  # Apply normalization
    return normalized.unsqueeze(0)  # Add batch dimension (B=1)

