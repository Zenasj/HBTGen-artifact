# torch.rand(B, 3, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 20, 3)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(20, 64, 3)),
                ('relu2', nn.ReLU()),
            ])
        )
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input with 3x32x32 image size (common for small models)
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

