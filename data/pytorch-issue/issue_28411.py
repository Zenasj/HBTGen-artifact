import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPPPooling

p = ASPPPooling(1, 1)
p.eval()

t = torch.zeros(1, 1, 3, 3)
print('ok:', p(t).shape)
# ok: torch.Size([1, 1, 3, 3])

pp = torch.jit.script(p)
print('bug:', pp(t).shape)
# bug: torch.Size([1, 1, 1, 1])

class ASPPPoolingV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingV2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2d = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.avg_pool(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

p = ASPPPoolingV2(1, 1)
p.eval()

print('ok:', p(t).shape)
# ok: torch.Size([1, 1, 3, 3])

pp = torch.jit.script(p)
print('ok:', pp(t).shape)
# ok: torch.Size([1, 1, 3, 3])