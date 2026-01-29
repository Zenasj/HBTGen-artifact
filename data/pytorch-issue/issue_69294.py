# torch.rand(4, 1, 64, 64, 64, dtype=torch.float32)  # 5D input shape (B, C, H, W, D)
import torch
from torch import nn, fft

class MyModel(nn.Module):
    def __init__(self, dim=(2, 3, 4)):
        super(MyModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        shifted = fft.fftshift(x, dim=self.dim)
        return fft.fftn(shifted, dim=self.dim)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1, 64, 64, 64, dtype=torch.float32).cuda()

