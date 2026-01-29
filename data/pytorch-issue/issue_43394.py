import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

# torch.rand(B, 3, H, W, dtype=torch.float)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 5
        kernel_size = 3
        stride = 1
        padding = (1, 2)  # Original padding parameter from the issue
        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding * 2, 0),  # Applies (1,2,1,2) padding
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        return self.dequant(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (B=1, C=3, H=64, W=64)
    return torch.rand(1, 3, 64, 64, dtype=torch.float)

