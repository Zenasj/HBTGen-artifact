# torch.rand(B, 3, 224, 338, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        H, W = x.shape[2], x.shape[3]
        lin = torch.linspace(0, 1, steps=H * W, device=x.device).view(1, 1, H, W)
        return x * lin.expand_as(x)  # Uses linspace in forward pass

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 338, dtype=torch.float32)

