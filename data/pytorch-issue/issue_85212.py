import torch

input = torch.full((64,), 1, dtype=torch.float32, requires_grad=False)
numel = 1250999896764
n_bins = 0
ratio = 0
bit_width = 0
torch.choose_qparams_optimized(input, numel, n_bins, ratio, bit_width)