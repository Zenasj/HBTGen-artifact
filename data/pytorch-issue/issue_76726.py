# torch.rand(B=7, C=1, H=10, W=20, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, 
                    out_channels=64, 
                    kernel_size=(8, 20), 
                    stride=(1, 1),
                    padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(7, 1, 10, 20, dtype=torch.float32)

