import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Inferred input shape for a small CNN
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN structure to demonstrate LR scheduler impact
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
    return MyModel()

def GetInput():
    # Random input matching the CNN's expected dimensions
    return torch.rand(1, 3, 32, 32, dtype=torch.float32).cuda()  # GPU usage for sync demonstration

