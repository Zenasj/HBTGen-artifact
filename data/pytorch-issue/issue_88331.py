# torch.rand(1, 4, 16, 16, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_in = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(32, 128, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = 5.5 * x
        out = self.conv_in(out)
        # Apply GroupNorm three times with residual connections
        out = out + self.norm(out)
        out = out + self.norm(out)
        out = out + self.norm(out)
        # Upsample spatial dimensions
        out = F.interpolate(out, scale_factor=8.0, mode='nearest')
        # Additional GroupNorm after interpolation
        out = self.norm(out)
        # Final convolution layer
        out = self.conv_out(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 16, 16, dtype=torch.float32)

