# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + identity

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(True)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)  # First upscale
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)  # 64 = 256/(2^2)
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # Second upscale
        self.conv4 = nn.Conv2d(64, 3, kernel_size=9, padding=4)  # Final output channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual_blocks(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pixel_shuffle2(x)
        x = self.conv4(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

