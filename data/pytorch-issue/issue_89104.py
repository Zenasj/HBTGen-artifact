import torch.nn as nn

import torch

x = torch.rand(1, 32, 512, 512, 256)
m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False)
y = m(x)