import torch

self = torch.full((1, 5, 3,), 2, dtype=torch.cfloat, requires_grad=False)
dim = [-1250999896764]
normalization = 0
last_dim_size = 0
torch._fft_c2r(self, dim, normalization, last_dim_size)