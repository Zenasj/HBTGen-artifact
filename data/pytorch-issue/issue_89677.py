import torch.nn as nn

import torch

self = torch.tensor([2, 4, 8, 8, 7, 3, 1, 8, 7, 8, 1, 1, 0, 0, 3, 1, 0, 2, 1, 5, 8, 7, 9, 7, 0, 0, 7, 6, 5, 5, 9, 1, 6, 2, 9, 1, 4, 3, 2, 3, 1, 1, 0, 6, 3, 9, 3, 9, 6, 6, 9, 2, 8, 5, 7])
weight = torch.rand([100, 5], dtype=torch.float32)
offsets = torch.tensor([0,  6, 12, 15, 25, 32, 40, 42, 46, 53, 53])

res = torch.nn.functional.embedding_bag(
    self, weight, offsets,
    norm_type = 2.0,
    scale_grad_by_freq = False,
    mode = 'mean',
    sparse = True,
    include_last_offset = True,
    padding_idx = 61
)

import torch

self = torch.tensor([2, 4, 8, 8, 7, 3, 1, 8, 7, 8, 1, 1, 0, 0, 3, 1, 0, 2, 1, 5, 8, 7, 9, 7, 0, 0, 7, 6, 5, 5, 9, 1, 6, 2, 9, 1, 4, 3, 2, 3, 1, 1, 0, 6, 3, 9, 3, 9, 6, 6, 9, 2, 8, 5, 7])
weight = torch.rand([100, 5], dtype=torch.float32)
offsets = torch.tensor([0,  6, 12, 15, 25, 32, 40, 42, 46, 53, 53])

res = torch.nn.functional.embedding_bag(
    self, weight, offsets,
    norm_type = 2.0,
    scale_grad_by_freq = False,
    mode = 'mean',
    sparse = True,
    include_last_offset = True,
    padding_idx = 61
)
print('still alive')