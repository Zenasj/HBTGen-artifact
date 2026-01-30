import torch.nn as nn

import torch

dtype = torch.cuda.FloatTensor

bad_sizes = [
  (1, 256, 109, 175),
  (1, 256, 80, 128),
  (1, 256, 120, 192),
]

x = torch.randn(*bad_sizes[2]).type(dtype)
x_var = torch.autograd.Variable(x)
print('x.size() = ', x.size())

conv = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
conv.type(dtype)

y = conv(x_var)
print(y.size())