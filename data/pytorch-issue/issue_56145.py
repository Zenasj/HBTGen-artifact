import torch.nn as nn

import torch
x1 = torch.randn(1, 1, 3, 3, requires_grad=True)
conv1 = torch.nn.Conv2d(1, 1, 3)
y1 = conv1(x1)
print(y1, y1.requires_grad)
# tensor([[[[0.3465]]]]) False

x2 = torch.randn(1, 2, 3, 3, requires_grad=True)
conv2 = torch.nn.Conv2d(2, 1, 3)
y2 = conv2(x2)
print(y2, y2.requires_grad)
# tensor([[[[0.7109]]]], grad_fn=<ThnnConv2DBackward>) True