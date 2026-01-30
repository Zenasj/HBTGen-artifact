import torch

self = torch.full((1, 51, 10,), 1, dtype=torch.float64, requires_grad=False)
dim = [-1250999896764]
normalization = 0
onesided = True
torch._fft_r2c(self, dim, normalization, onesided)