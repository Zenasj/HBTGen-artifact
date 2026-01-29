# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = SubModule()
        self.sub2 = SubModule()
        self.fc = nn.Linear(64, 10)  # Concatenated output from sub1 and sub2 (32+32)

    def forward(self, x):
        x1 = self.sub1(x)
        x2 = self.sub2(x)
        x = torch.cat((x1, x2), dim=1)  # Concatenate along feature dimension
        return self.fc(x)

class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, 32)  # After global pooling

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

