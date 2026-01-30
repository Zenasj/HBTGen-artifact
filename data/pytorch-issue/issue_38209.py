import torch
x = torch.randn((10, 2)).cuda()
x.requires_grad = True
y = x*x
l = y.argmax(-1)
z = y[torch.arange(y.shape[0]), l]
z.sum().backward()