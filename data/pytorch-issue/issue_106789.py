import torch

x = torch.tensor([2., 3, 0, 0], requires_grad=True)
y = torch.prod(x)
gx, = torch.autograd.grad(y.sum(), x, create_graph=True)