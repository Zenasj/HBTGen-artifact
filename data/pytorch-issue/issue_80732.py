import torch.nn as nn

import torch

avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
a = torch.randn(64, 512, 4, 4, device='mps')
avgpool(a)