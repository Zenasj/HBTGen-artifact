import torch

input = torch.zeros(1, 4, dtype=torch.complex64)
dim = (-1,)   # RuntimeError
normalization = 0
onesided = True
torch._fft_c2c(input, dim, normalization, onesided)

import torch

input = torch.zeros(1, 4, dtype=torch.complex64)
dim = (1,)  # RuntimeError
normalization = 0
last_dim_size = 0
torch._fft_c2r(input, dim, normalization, last_dim_size)

import torch

input = torch.zeros(1, 4, dtype=torch.complex64)
dim = (1,)  # RuntimeError
normalization = 0
last_dim_size = 0
torch._fft_c2r(input, dim, normalization, last_dim_size)