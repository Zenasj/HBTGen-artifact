import torch.nn as nn

import torch
channels = 320
hw = 1 # also can be 10 or other int get similar result
x = torch.rand(channels, channels, hw, hw, dtype=torch.float32, requires_grad=False).cuda()

a = torch.nn.Conv2d(channels, channels, 5, padding=2, bias=False, groups=channels).eval().cuda()
b = torch.nn.Conv2d(channels, channels, 5, padding=2, bias=False, groups=channels).eval().cuda()

y1 = torch.maximum(x, a(x))
y2 = x+torch.nn.functional.relu(a(x)-x)
y3 = a(x)+torch.nn.functional.relu(x-a(x))

print(torch.sum(y1-y2))
print(torch.sum(y1-y3))