import torch.nn as nn

import torch
m = torch.nn.GroupNorm(5, 6)
x = torch.randn(1, 6, 4, 4)
y = m(x)