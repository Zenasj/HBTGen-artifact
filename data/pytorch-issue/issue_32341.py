# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)
        self.channel_c = nn.Conv2d(192, 128, 1)
        self.l = nn.GroupNorm(32, 128)

    def forward(self, x):
        x = self.expander(x)
        x = self.channel_c(x)
        x = self.l(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 544, 1920)

