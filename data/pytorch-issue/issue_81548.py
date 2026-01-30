import torch
import torch.nn as nn

import time
from torch import nn

class EasyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EasyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.relu3 = nn.ReLU6(inplace=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.relu4 = nn.ReLU6(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.derelu1 = nn.ReLU6(inplace=True)
    
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.relu5 = nn.ReLU6(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.model = nn.Sequential(self.conv1, self.relu1, self.conv2,  self.relu2, self.conv3, self.relu3, self.conv4,  self.relu4, self.deconv1, self.derelu1,
                                    self.conv5,  self.relu5, self.avgpool)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def main():
    model = EasyModel()
    model.cuda()
    x = torch.randn(2,3,128,128).cuda()
    y = model(x)
    after_forward = torch.cuda.memory_allocated()
    print("after_forward: ", after_forward)

if __name__ == "__main__":
    main()