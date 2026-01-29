# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use fixed 1x1 convolution with 3 input/output channels (matches test input)
        self.conv = nn.Conv2d(3, 3, kernel_size=1)  # Static weight for ONNX compatibility

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)  # Force float32 as in original code
        out = F.pad(x, (1, 1, 1, 1), "constant", 0)
        # Use predefined convolution layer instead of dynamic weight
        out = self.conv(out)
        return out.to(dtype)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches original test input dimensions (B=1, C=3, H=224, W=224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

