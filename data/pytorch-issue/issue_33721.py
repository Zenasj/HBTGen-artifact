# torch.rand(B, 3, 128, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 64 * 64, 100)  # Adjusted for pooling dimensions

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # Added pooling to reduce spatial dimensions
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 64  # Matches DataLoader's batch_size=64
    return torch.rand(B, 3, 128, 128, dtype=torch.float32)

