import torch.nn as nn

import torch

input = torch.randint(-128, 128, [3, 6], dtype=torch.int64)
weight = torch.rand([30522, 384], dtype=torch.float32)
padding_idx = 0
max_norm = None
norm_type = 2.0
scale_grad_by_freq = False
sparse = False
torch.nn.functional.embedding_bag(
    input,
    weight,
    max_norm=max_norm,
    norm_type=norm_type,
    scale_grad_by_freq=scale_grad_by_freq,
    sparse=sparse,
    padding_idx=padding_idx,
)

import torch

input = torch.randint(-128, -1, [3, 6], dtype=torch.int64)
weight = torch.rand([30522, 384], dtype=torch.float32)
padding_idx = 0
max_norm = None
norm_type = 2.0
scale_grad_by_freq = False
sparse = False
torch.nn.functional.embedding_bag(
    input,
    weight,
    max_norm=max_norm,
    norm_type=norm_type,
    scale_grad_by_freq=scale_grad_by_freq,
    sparse=sparse,
    padding_idx=padding_idx,
)