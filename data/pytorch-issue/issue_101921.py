# torch.rand(B, C, H, W, dtype=torch.float32)  # B: batch size, C: channels, H: height, W: width

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 全卷积层
        self.fc1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        # 卷积层
        x = cp.checkpoint(self.conv1, x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = cp.checkpoint(self.conv2, x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        x = cp.checkpoint(self.conv3, x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, 2)

        # 全卷积层
        x = cp.checkpoint(self.fc1, x)
        x = nn.functional.relu(x, inplace=True)
        x = cp.checkpoint(self.fc2, x)
        x = nn.functional.relu(x, inplace=True)
        # x = self.fc3(x)

        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    B, C, H, W = 8, 3, 448, 448  # Example dimensions
    images = torch.randn((B, C, H, W), dtype=torch.float32)
    return images

