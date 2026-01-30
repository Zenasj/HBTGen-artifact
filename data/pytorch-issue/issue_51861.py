import torch
a = torch.rand(10)
a.requires_grad = True
a.sum().backward()