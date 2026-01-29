# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for a typical image model
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)  # 224 -> 112 -> 56 after pooling
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns a basic CNN model with parameters for FSDP testing
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape (B=2, C=3, H=224, W=224)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

