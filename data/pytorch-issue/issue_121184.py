import torch.nn as nn

py
import torch
import math

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        t1 = self.conv(x)
        self.bn.eval()
        self.bn.running_mean.zero_()
        self.bn.running_var.fill_(1)
        return self.bn(t1)
func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)
func = Model()

with torch.no_grad():
    print(func(x.clone()))

    func1 = torch.compile(func)
    print(func1(x.clone()))