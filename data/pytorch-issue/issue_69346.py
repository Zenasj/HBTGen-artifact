# torch.rand(4, 32, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_features, init_size=None):
        super(MyModel, self).__init__()
        if init_size is None:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(size=init_size * 2, mode='bilinear', align_corners=True)
        self.innorm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x):
        out = self.upsample(x)
        out = self.innorm(out)
        return out

def my_model_function():
    # Use init_size=64 to force static output size for ONNX compatibility
    return MyModel(num_features=32, init_size=64)

def GetInput():
    # Matches input dimensions expected by MyModel with init_size=64
    return torch.randn(4, 32, 64, 64, dtype=torch.float32)

