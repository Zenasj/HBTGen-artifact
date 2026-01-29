# torch.rand(1, 1, 10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate NHWC tensor and permute to NCHW (non-contiguous)
    data = torch.rand(1, 10, 10, 1, dtype=torch.float32)
    x = data.permute(0, 3, 1, 2)  # Shape becomes (1, 1, 10, 10)
    return x

