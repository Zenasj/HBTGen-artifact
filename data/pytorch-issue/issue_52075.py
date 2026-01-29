# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape based on common image processing tasks
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 56 * 56, 10)  # 56x56 from 224/2^2 pooling

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns an instance with default initialization
    return MyModel()

def GetInput():
    # Generates random input matching expected shape (B=1 for simplicity)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

