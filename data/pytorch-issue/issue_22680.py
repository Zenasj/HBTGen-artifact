import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FuckNet2(nn.Module):

    def __init__(self):
        super(FuckNet2, self).__init__()

        self.welcome_layer = BasicConv(3, 256, 3, 1, 1)
        self.block_1 = BasicConv(256, 256, 3, 1, 1)
        self.fuck_layers = nn.Sequential()
        for i in range(5):
            self.fuck_layers.add_module('{}'.format(i), self.block_1)

    def forward(self, x):
        x = self.welcome_layer(x)
        return self.fuck_layers(x)

with torch.jit._enable_recursive_script(): 
    script_model = torch.jit.script(FuckNet2())