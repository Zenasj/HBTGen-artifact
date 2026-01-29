# torch.rand(128, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 56 * 56, 10)  # 224/2/2 = 56

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 112x112
        x = self.pool(self.relu(self.conv2(x)))  # 56x56
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(128, 3, 224, 224, dtype=torch.float32)

