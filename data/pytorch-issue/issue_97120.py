import torch
import torch.nn as nn

class Mod(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        res1 = self.relu(self.conv2(x1) + self.conv3(x1))
        res2 = self.relu2(self.conv4(res1) + res1)
        return res2