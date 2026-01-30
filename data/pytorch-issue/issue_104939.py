import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(TestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8, stride=4, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=6)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2, bias=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.conv3 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1, groups=3)
        self.relu3 = torch.nn.ReLU6()
        self.conv4 = torch.nn.LogSoftmax()
        self.relu4 = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        return x