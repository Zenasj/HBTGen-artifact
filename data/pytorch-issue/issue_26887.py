# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, bias=False)
        self.inner32 = nn.Conv2d(2048, 256, 1, padding=0, stride=1)
        self.inner16 = nn.Conv2d(1024, 256, 1, padding=0, stride=1)
        self.outer16 = nn.Conv2d(256, 256, 3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        feat32 = self.inner32(feat32)
        feat16 = self.inner16(feat16)
        _, _, H, W = feat16.size()
        feat32 = F.interpolate(feat32, (H, W), mode='nearest')
        feat16 = feat16 + feat32
        return feat4, feat8, feat16, feat32

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

