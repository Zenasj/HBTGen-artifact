# torch.rand(B, 8, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=8,
            out_channels=128,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 8, 64, dtype=torch.float32)

