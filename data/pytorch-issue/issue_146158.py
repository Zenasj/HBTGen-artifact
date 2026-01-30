import torch
t = torch.rand(10, 10)
out = torch.randn(10, 6, 6)
rfft2 = torch.fft.rfft2(t, out=out)