import torch.nn as nn

import torch

torch.backends.cudnn.enabled=False
x = torch.rand(1, 32, 512, 512, 256).to('cuda:0')
m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0,stride=1,bias=False).to('cuda:0')
x = m(x)  # Assert!!