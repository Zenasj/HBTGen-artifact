# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=(1, 1), padding=1)
    
    def forward(self, x):
        return self.conv1(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 32
    channels = 3
    height = 256
    width = 256
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32)

