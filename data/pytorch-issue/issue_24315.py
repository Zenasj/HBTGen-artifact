import torch
import torch.nn as nn

m = torch.nn.Conv2d(4, 4, 3, groups=4)
torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')