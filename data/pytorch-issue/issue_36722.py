import torch

a = torch.rand([], requires_grad=True, device='cuda:0')
b = torch.rand(10, requires_grad=True, device='cuda:1')

c = a * b
torch.cuda.synchronize()

0x7f4e87400400
0x7f4e87400200
0x7f4e87400000

import torch

a = torch.rand([], requires_grad=True, device='cuda:0')
b = torch.rand(10, requires_grad=True, device='cuda:1')

c = a.to('cuda:1') * b
torch.cuda.synchronize()