# torch.rand(B=4, C=4, H=2, W=2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, 2, 2, dtype=torch.float32)

