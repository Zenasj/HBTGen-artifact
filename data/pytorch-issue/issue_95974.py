import torch

a = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.linalg.solve(a, b)
torch.autograd.grad(c.sum(), a)
# ^^ This line blows up