# torch.rand(8, 64, 128, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel().cuda()

def GetInput():
    x = torch.randn(8, 64, 128, 128).cuda()
    x[:] = x[0]  # Ensure all batch elements are identical
    return x

