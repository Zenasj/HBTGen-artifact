# torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.bfloat16)
        self.bn = nn.BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

