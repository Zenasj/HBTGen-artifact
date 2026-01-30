import random

import torch
torch.random.manual_seed(1)
N=4
b = torch.randn([N, N], device='cpu')
print(b @ b.t())
torch.mm(b, b.t(), out=b)
print(b)

import torch
torch.random.manual_seed(1)
N=5000
b = torch.randn([N, N], device='cuda')
print(torch.any(torch.isnan(b @ b.t())))
torch.mm(b, b.t(), out=b)
print(torch.any(torch.isnan(b)))