# torch.rand(B, 4, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.shape
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, height, width)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.shuffle = ChannelShuffle(groups=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle(x)
        x = self.conv2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4, 32, 32, dtype=torch.float32)

