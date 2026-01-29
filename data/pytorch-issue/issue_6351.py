# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Inferred input shape for image classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture to demonstrate deterministic training issues
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns random input tensor matching the model's expected dimensions
    return torch.rand(4, 3, 32, 32, dtype=torch.float32)  # Batch size 4, 32x32 RGB images

