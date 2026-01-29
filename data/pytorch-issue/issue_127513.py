# torch.rand(B, C, H, W, dtype=torch.float32)  # Shape: [1, 3, 224, 224]

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(3)
    
    def forward(self, x):
        return self.bn(self.conv(x))

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape [1, 3, 224, 224]
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

