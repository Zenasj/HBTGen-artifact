import torch

n = 10
out = torch.empty(2, 2, requires_grad=True).clone()
freq = torch.fft.rfftfreq(n, out=out)