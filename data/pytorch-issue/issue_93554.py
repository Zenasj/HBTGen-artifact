# torch.rand(3, 6, 10, 10, dtype=torch.float32).to(memory_format=torch.channels_last)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 3, 3)
        self.bn = nn.BatchNorm2d(3, eps=0.001)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 6, 10, 10, dtype=torch.float32).to(memory_format=torch.channels_last)

