import torch
x = torch.rand(3, 3)
x.long().requires_grad_()  # will throw an RuntimeError
x.requires_grad_().long()  # will work and get an Long tensor with requires_grad=True