import torch.nn as nn

import torch
a = torch.rand((1,1,2), device='cuda')
b = torch.rand((1,1,2,3,1), device='cuda')
def forward():
    c = torch.nn.functional.pad(a, (0, 1, 0, 0), 'reflect') # [1,1,3]
    d = torch.add(b, c) # [1,1,2,3,1] + [1,1,3] -> [1,1,2,3,3]
    return torch.nn.functional.pad(d, (-2, 0, 0, 0, 0, 0, 0, 0, 0, 1))
fn_compiled = torch.compile(forward)
print(fn_compiled())

import torch

a = torch.rand([2,2,1,1,1], device='cuda')
b = torch.rand([2,2,2], device='cuda')
def forward():
    c = torch.add(a, b)[:, 0:1] + 1
    d = c.flatten()
    d[1] = 0.
    return torch.relu(c.sum(0))

with torch.no_grad():
	print(forward())
	fn_compiled = torch.compile(forward)
	print(fn_compiled())