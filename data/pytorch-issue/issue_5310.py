# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        kernel_size = (inp_size[2], inp_size[3])  # Use full spatial dimensions as kernel
        return F.max_pool2d(x, kernel_size=kernel_size)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = MyAdaptiveMaxPool2d()  # Custom replacement for AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 3, 224, 224  # Example input dimensions (BATCH, CHANNELS, HEIGHT, WIDTH)
    return torch.rand(B, C, H, W, dtype=torch.float32)

