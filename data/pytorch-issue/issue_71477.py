import torch
input = torch.rand([], dtype=torch.float64)
dim = 100
torch.cummin(input, dim)
# torch.return_types.cummin(
# values=tensor(0.8172, dtype=torch.float64),
# indices=tensor(0))

import torch
input = torch.rand([1], dtype=torch.float64)
dim = 100
torch.cummin(input, dim)
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 100)