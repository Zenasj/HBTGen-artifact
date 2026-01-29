# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (1, 3, 512, 512)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyResNet(nn.Module):
    def __init__(self):
        super(DummyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # feat8: 64 channels, 1/2 spatial
        feat8 = x
        x = F.relu(self.conv2(x))  # feat16: 128 channels, 1/4 spatial
        feat16 = x
        x = F.relu(self.conv3(x))  # intermediate: 256 channels, 1/8 spatial
        x = F.relu(self.conv4(x))  # feat32: 512 channels, 1/16 spatial
        feat32 = x
        return feat8, feat16, feat32  # Matches original forward structure

class MyModel(nn.Module):
    def __init__(self, num_classes=19):
        super(MyModel, self).__init__()
        self.resnet = DummyResNet()
        self.conv_avg = nn.Conv2d(512, 128, kernel_size=1)  # Matches error's weight shape (128x512)
        self.arm32 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv_head32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.arm16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_head16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x):
        H0, W0 = x.size(2), x.size(3)
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size(2), feat8.size(3)
        H16, W16 = feat16.size(2), feat16.size(3)
        H32, W32 = feat32.size(2), feat32.size(3)
        
        # Fixed avg_pool2d with static kernel_size (ONNX-compliant)
        size_array = [int(s) for s in feat32.size()[2:]]
        avg = F.avg_pool2d(feat32, kernel_size=size_array)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 512, 512, dtype=torch.float32)

