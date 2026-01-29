# torch.rand(2, 3, 224, 224, dtype=torch.float32)  # Inferred input shape from common CNN use cases
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture to match CUDA profiling scenario
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # 53 derived from 224 → 110 → 53 after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    # Returns a basic CNN instance
    return MyModel()

def GetInput():
    # Returns random tensor matching input shape (B=2, C=3, H=224, W=224)
    return torch.randn(2, 3, 224, 224, dtype=torch.float32)

