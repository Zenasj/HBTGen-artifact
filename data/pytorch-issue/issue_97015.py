import torch

x = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
a = torch.tensor(1, dtype=torch.float32, requires_grad=True)
y = x * a
dydx = torch.autograd.grad(y, x, create_graph=True, allow_unused=True)
d2ydx2 = torch.autograd.grad(dydx, x, allow_unused=True, zero_grad_unused=True)
try:
    d3ydx3 = torch.autograd.grad(d2ydx2, x, allow_unused=True, zero_grad_unused=True)
except RuntimeError as e:
    assert False, "Should not raise error"