# torch.rand(B, 1, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        return x

def my_model_function():
    # Initialize with num_classes=1 as in original test case
    return MyModel(num_classes=1)

def GetInput():
    # Matches the input shape (B=1, C=1, H=28, W=28)
    return torch.randn(1, 1, 28, 28, dtype=torch.float32)

