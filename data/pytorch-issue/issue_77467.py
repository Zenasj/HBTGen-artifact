# torch.rand(B, 3, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))

def my_model_function():
    return MyModel()

def GetInput():
    # Example dimensions matching common image input (batch, channels, height, width)
    B, C, H, W = 2, 3, 32, 32
    return torch.rand(B, C, H, W, dtype=torch.float32)

