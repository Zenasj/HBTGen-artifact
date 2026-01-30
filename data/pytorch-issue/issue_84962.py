import torch.nn as nn

import torch
import torch.nn
bn = torch.nn.BatchNorm2d(3,affine=False)
x = torch.randn(20, 3, 35, 45)
y = bn(x)
bn2 = bn.to("mps")
x2 = x.to("mps")
y2 = bn2(x2)
y2 = y2.to("cpu")
print (torch.max(torch.abs(y-y2)))
print(torch.max(y),torch.max(y2))