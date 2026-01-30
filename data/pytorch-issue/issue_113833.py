import numpy as np
import torch
import torch.nn as nn

x = torch.rand(100, 50, 512, 512).cuda()
c = nn.Conv2d(50, 200, kernel_size=1).cuda()
r = nn.ReLU().cuda()

loss = torch.sum(r(c(x)))
print(loss)
loss.backward() # this is fine though

loss = torch.prod(r(c(x)))
print(loss)
loss.backward() # this line gives you the error