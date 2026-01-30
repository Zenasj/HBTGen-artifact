import torch.nn as nn

import torch
import time

layer = torch.nn.Conv2d(3, 1, 3)
inp = torch.rand(10, 3, 1000, 1000)

out = layer(inp)
c = 5

res = out.unfold(3, c, 1)
res.sum().backward()