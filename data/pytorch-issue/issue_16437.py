import torch.nn as nn

import torch
from torch import nn

torch.backends.cudnn.deterministic = True

x = torch.randn(1, 1, 16, 16).to('cuda').half()
conv = nn.Conv2d(1, 1, 3, bias=False).to('cuda').half()

print(conv(x).shape)