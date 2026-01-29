# torch.rand(B, 8, 4, 4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=8, out_channels=1024, kernel_size=(3, 3, 3), bias=False, padding=(1, 1, 1))
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input_shape=8 4 4 4 and batch_size=4 from the original reproduction steps
    return torch.rand(4, 8, 4, 4, 4, dtype=torch.float32)

