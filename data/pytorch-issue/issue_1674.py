import torch

net.zero_grad()
out.backward(torch.randn(1, 10))