# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv3d = nn.Conv3d(8, 8, (1, 1, 1), stride=(1, 1, 1), padding=(1, 1, 1), groups=8, dilation=(1, 1, 1), bias=True)

    def forward(self, x):
        return self.conv3d(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 8, 1, 10, 10, dtype=torch.float32)

