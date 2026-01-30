import torch
print(torch.__version__)
out = torch.fft.rfft(torch.randn(1000).cuda())
print(out.sum())