import torch

input = torch.full([7, 7, 4, 6, 3, 1, 2], -7124, dtype=torch.float32, requires_grad=False)
numel = -8192
n_bins = 1255
ratio = -9185
bit_width = -4519
torch.choose_qparams_optimized(input, numel, n_bins, ratio, bit_width)