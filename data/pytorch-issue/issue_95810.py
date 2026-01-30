import torch.nn as nn

import torch
from torch import nn

device = 'cuda:0'
dtype = torch.float64 # double for gradcheck
ndim = 128 # test should pass for < 128
a = torch.zeros((ndim, 1), device=device, dtype=dtype)
a.requires_grad=True
ln = nn.LayerNorm(1).to(device).to(dtype)
torch.autograd.gradcheck(ln, a)
torch.cuda.synchronize()
print('gradcheck passed')
torch.autograd.grad(ln(a).sum(), ln.parameters())
torch.cuda.synchronize()
print('total derivative passed')
torch.autograd.grad(ln(a).sum(), ln.weight)
torch.cuda.synchronize()
print('partial derivative passed')