import torch

a = torch.rand(10, requires_grad=True)

b = a.repeat(0)

b.sum().backward()