# Inputs: (torch.rand(B, C, H, W, dtype=torch.float32), torch.rand(B, C, D, H, W, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg2d = nn.AdaptiveAvgPool2d([-36, 0])  # Problematic output_size
        self.avg3d = nn.AdaptiveAvgPool3d([-36, 0, 0])  # Problematic output_size
        self.max3d = nn.AdaptiveMaxPool3d((-5, 0, 0))  # Should throw an error

    def forward(self, inputs):
        x2d, x3d = inputs

        avg2d_ok = True
        try:
            _ = self.avg2d(x2d)
        except:
            avg2d_ok = False

        avg3d_ok = True
        try:
            _ = self.avg3d(x3d)
        except:
            avg3d_ok = False

        max3d_ok = False  # True means error was thrown (desired behavior)
        try:
            _ = self.max3d(x3d)
        except RuntimeError:
            max3d_ok = True

        # Bug exists if avg2d/avg3d do not throw (ok) and max3d does throw (ok)
        bug_exists = avg2d_ok and avg3d_ok and max3d_ok
        return torch.tensor([1.0 if bug_exists else 0.0], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # 4D input for 2D AdaptiveAvgPool
    x2d = torch.rand(128, 2048, 4, 4, dtype=torch.float32)
    # 5D input for 3D AdaptiveAvgPool and AdaptiveMaxPool
    x3d = torch.rand(4, 4, 128, 2048, 4, dtype=torch.float32)
    return (x2d, x3d)

