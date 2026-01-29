import torch
from torch import nn

# torch.rand(1, 1, 1, 1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, im, **kwargs):
        return self.conv(im)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1)

