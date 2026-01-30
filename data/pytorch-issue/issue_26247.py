import torch
a = torch.tensor([True, False], device='cuda')
res = torch.where(a > 0)
print(res)


a = torch.tensor([True, False], device='cpu')
res = torch.where(a > 0)
print(res)

import torch
a = torch.tensor([True, False], device='cuda')
x = torch.tensor([True, True], device='cuda')
y = torch.tensor([False, False], device='cuda')
res = torch.where(a, x, y)

import torch
a = torch.tensor([True, False], device='cpu')
x = torch.tensor([True, True], device='cpu')
y = torch.tensor([False, False], device='cpu')
res = torch.where(a, x, y)

import torch
a = torch.tensor([True, False], device='cpu')
x = torch.tensor([True, True], device='cpu')
y = torch.tensor([False, False], device='cpu')
res = torch.where(a, x, y)

import torch
a = torch.tensor([True, False], device='cuda')
x = torch.tensor([True, True], device='cuda')
y = torch.tensor([False, False], device='cuda')
res = torch.where(a, x, y)