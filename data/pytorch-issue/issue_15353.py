import torch.nn as nn

import torch
from torch import nn, autograd

m = nn.Conv2d(2, 3, 3)

x = torch.rand(1,2,4,4, requires_grad=True)
y = m(x)

g, = autograd.grad(y.sum(), x, create_graph=True)
print(g.requires_grad)