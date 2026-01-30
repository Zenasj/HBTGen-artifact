import torch
x = torch.randn(3, 3, requires_grad=True)
del x.grad