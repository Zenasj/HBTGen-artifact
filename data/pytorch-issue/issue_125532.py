import torch

x1 = torch.rand((10, 10), requires_grad=True)
x2 = torch.rand((10, 10), requires_grad=True)

y = x1 * x2
dydx1 = torch.autograd.grad(y, x1, torch.ones_like(y))