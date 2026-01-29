# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
        self.norm = nn.InstanceNorm2d(32, track_running_stats=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.relu()
        x = self.norm(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 32, 8, 8, dtype=torch.float32)

