# torch.rand(16, 3, 224, 224, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3, affine=True, track_running_stats=True)
    
    def forward(self, x):
        return self.bn(x)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(16, 3, 224, 224, dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_()
    return x

