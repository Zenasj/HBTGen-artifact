# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        return x[:, :, :, None, :, None].expand(
            -1, -1, -1, self.upscale, -1, self.upscale
        ).reshape(
            x.size(0),
            x.size(1),
            x.size(2)*self.upscale,
            x.size(3)*self.upscale
        )

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            UpsampleDeterministic(2),
            nn.Conv2d(64, 64, 3, 1, 1),
            UpsampleDeterministic(2),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.seq(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

