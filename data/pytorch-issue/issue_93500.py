import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)


    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv2(x)
        y = x2 * x
        return self.pool(y)