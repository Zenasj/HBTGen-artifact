import torch.nn as nn

import torch

inC = 4
outC = 16

x = torch.zeros(1, inC, 1, 1)
x[0,3,0,0] = 1.0
print("x", x)

weight = torch.ones(outC, inC//2, 1, 1)

y1 = torch.nn.functional.conv2d(x, weight, groups=2)
print("y1", y1)
y2 = torch.nn.functional.conv2d(x.cuda(), weight.cuda(), groups=2).cpu()
print("y2", y2)