# torch.rand(B, C, H, W, dtype=...)  # B: batch size, C: channels, H: height, W: width

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, out_size: int = 32):
        super().__init__()

        self.c2 = nn.Conv2d(out_size, 4 * out_size, 4, stride=4)
        self.u2 = nn.ConvTranspose2d(4 * out_size, out_size, 4, stride=4)

        self.merge = nn.Conv2d(2 * out_size, out_size, 1)

    def forward(self, X):
        Z = self.u2(self.c2(X))
        return self.merge(torch.cat([X, Z], dim=1))

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    # The input shape is (B, C, H, W) where B is the batch size, C is the number of channels,
    # and H, W are the height and width of the input image.
    # For this example, we use B=192, C=32, H=256, W=256.
    return torch.randn(192, 32, 256, 256, dtype=torch.float32).cuda()

