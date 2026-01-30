py
import torch

def fn(input):
    fn_res = input.xlogy(2)
    return fn_res

input = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input))

[
[0.3010, 0.0000, 0.0000, 0.0000],
[0.0000, 0.3010, 0.0000, 0.0000],
[0.0000, 0.0000, 0.3010, 0.0000],
[0.0000, 0.0000, 0.0000, 0.3010]
]

import torch

def fn(input):
    t = torch.tensor([[3., 6., 0, 3.]], dtype=torch.float64, requires_grad=True)
    fn_res = input.xlogy(t)
    return fn_res

input = torch.tensor([[2., 0., 4., 0.]], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input))

import torch

x = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[0., 1., -1., -6.]], dtype=torch.float64, requires_grad=True)
z = torch.special.xlogy(x, 1. + y)
z.sum().backward()
print('xlogy(x, 1 + y):', x.grad)

x = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[0., 1., -1., -6.]], dtype=torch.float64, requires_grad=True)
z = x * torch.log(1. + y)
z.sum().backward()
print('x * log(1 + y): ', x.grad)

x = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[0., 1., -1., -6.]], dtype=torch.float64, requires_grad=True)
z = torch.special.xlog1py(x, y)
z.sum().backward()
print('xlog1py(x, y):  ', x.grad)

x = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[0., 1., -1., -6.]], dtype=torch.float64, requires_grad=True)
z = x * torch.log1p(y)
z.sum().backward()
print('x * log1p(y):   ', x.grad)