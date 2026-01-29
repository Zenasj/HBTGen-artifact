# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
from torch.ao.quantization.fuse_modules import fuse_modules

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    m = MyModel().eval()
    modules_to_fuse = ['conv1', 'bn1']
    fused_m = fuse_modules(m, modules_to_fuse)
    return fused_m

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Assuming a common input shape for image processing
    return torch.rand(B, C, H, W, dtype=torch.float32)

