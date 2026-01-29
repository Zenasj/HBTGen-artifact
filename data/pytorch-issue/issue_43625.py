# torch.rand(B, 1, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet5ImprovedOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.max_pool_2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(400, 180)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(180, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class LeNet5ImprovedQuantized(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.max_pool_2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.quantized.dynamic.Linear(400, 180)  # DynamicQuantizedLinear
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.quantized.dynamic.Linear(180, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.quantized.dynamic.Linear(100, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_model = LeNet5ImprovedOriginal()
        self.quantized_model = LeNet5ImprovedQuantized()

    def forward(self, x):
        return self.original_model(x)  # Return original output; quantized is encapsulated for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 28, 28, dtype=torch.float32)

