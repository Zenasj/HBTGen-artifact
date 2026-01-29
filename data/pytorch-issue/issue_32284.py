# torch.rand(B, C, H, W, dtype=...)  # (2, 1, 80, 80)  # Inferred input shape from the issue

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(64 * 20 * 20, 6)

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    batch = 2
    channels = 1
    side_dim = 80
    return torch.randn([batch, channels, side_dim, side_dim])

