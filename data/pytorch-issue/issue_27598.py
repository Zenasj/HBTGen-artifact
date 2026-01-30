import torch
import torch.nn as nn

input = torch.randn(1, 1, 32, 32)
c = nn.Conv2d(1, 32, 3, stride=0, bias=False, padding=(0, 1), padding_mode='constant')
c(input)