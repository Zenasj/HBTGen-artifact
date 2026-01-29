# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Input shape inferred from example

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)  # 224/2 (after pool) → 112 → /2 again → 56
        self.fc2 = nn.Linear(256, 10)  # Example output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns a model instance configured for oneDNN Graph + BF16
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape (B=1, C=3, H=224, W=224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

