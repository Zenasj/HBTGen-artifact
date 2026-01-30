import torch

x = torch.rand(10,10,10,requires_grad=True)
x = torch.fft.fftn(x)
x.mean().backward()