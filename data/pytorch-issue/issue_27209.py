import torch

n = 102  # Don't fail for n < 102

x = torch.randn(n, 1).to("cuda:0")
x.requires_grad = True

dist = torch.cdist(x, x, p=2)
dist.sum().backward()

dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0), p=2)