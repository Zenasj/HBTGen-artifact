# torch.rand(B, 1, 32, 32, dtype=torch.float32)  # Input shape: (batch, channels=1, height=32, width=32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 4D tensor with 1 channel, 32x32 image, and arbitrary batch size (default: 1)
    return torch.rand(1, 1, 32, 32, dtype=torch.float32)

