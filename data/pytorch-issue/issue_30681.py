# torch.rand(1, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.upsample_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_nearest = nn.UpsamplingNearest2d(scale_factor=2)
        self.interpolate = nn.Upsample(scale_factor=2, mode='nearest')  # Matches F.interpolate's default mode

    def forward(self, x):
        out_bilinear = self.upsample_bilinear(x)
        out_nearest = self.upsample_nearest(x)
        out_interpolate = self.interpolate(x)
        return (out_bilinear, out_nearest, out_interpolate)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

