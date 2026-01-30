import torch.nn as nn

import torch
x = torch.rand(1, 16, 2*70000, 10).to('cuda:0')
print('size of tensor: {:,} bytes'.format(x.element_size()*x.nelement()))
print('nelement: {}'.format(x.nelement()))
conv = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), bias=False).to('cuda:0')
out = conv(x)
print(out.shape)