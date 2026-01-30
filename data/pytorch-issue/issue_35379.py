import torch

x = torch.randn(3, requires_grad=True).cuda()
y = x ** 2
y.sum().backward()
x.grad