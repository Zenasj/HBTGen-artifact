# torch.rand(1, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Based on FAN (Facial Alignment Network) architecture leading up to problematic interpolate step
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Critical interpolate step causing discrepancy between PyTorch versions
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.residual_block(x) + x  # Example residual connection
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

