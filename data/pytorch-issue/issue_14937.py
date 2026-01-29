# torch.rand(1, 3, 256, 256, dtype=torch.float32)  # Inferred input shape from dummy_input

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

