# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.conv14 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.conv15 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )
        self.gap = nn.AdaptiveAvgPool3d(output_size=1)
        self.classifier = nn.Conv3d(512, 10, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.gap(x)
        out = self.classifier(x).squeeze()
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 3, 128, 128, 128, dtype=torch.float32)

