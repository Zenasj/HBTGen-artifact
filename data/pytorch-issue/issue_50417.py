import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same' size (inferred from common YOLOv5 practices)
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    # Standard convolution (copied from issue's code)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Minimal YOLOv5-like structure using the provided Conv class
        # Input shape: (B, 3, 640, 640) as typical for YOLOv5
        self.layer1 = Conv(3, 64, k=3, s=1)  # Example starting layer
        self.layer2 = Conv(64, 128, k=3, s=2)  # Example subsequent layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with QAT-ready configuration
    model = MyModel()
    # Apply QAT preparation (simplified for code completeness)
    # Note: Actual QAT setup would require prepare_qat(), but omitted here
    return model

def GetInput():
    # Return float32 tensor matching expected input shape (B, 3, 640, 640)
    return torch.rand(1, 3, 640, 640, dtype=torch.float32)

