python
import torch

nbatch = 10
nvec = 1

x = torch.ones((nbatch, nvec))
p = torch.ones((nbatch,1,1), requires_grad = True)
g = torch.ones_like(x)

A = torch.rand((nbatch,nvec,nvec)) * p
y = torch.linalg.solve(A, x)

value = torch.autograd.grad(y, [p], g)