import torch

tensor_0 = torch.full((8, 8, 0, 0, 0,), 1.5e+300, dtype=torch.float64, requires_grad=False)
tensor_1 = torch.full((8, 8, 0, 0, 0,), 1, dtype=torch.int64, requires_grad=False)
tensor_2 = torch.full((8, 8, 0, 0, 0, 14, 13, 0, 0, 6,), 1, dtype=torch.int64, requires_grad=False)
bool_3 = True
int_4 = 0
bool_5 = True
tensor_6 = torch.full((4,), 1, dtype=torch.int64, requires_grad=False)
bool_7 = True
int_8 = 0
torch.embedding_bag(tensor_0, tensor_1, tensor_2, bool_3, int_4, bool_5, tensor_6, bool_7)

import torch

weight = torch.full((2, 0, 0, 6, 6,), 0, dtype=torch.float64, requires_grad=False)
indices = torch.full((2, 0, 0, 6, 6,), 2, dtype=torch.int64, requires_grad=False)
offsets = torch.full((2, 0, 0, 6, 6, 8, 6, 8, 0, 6, 0, 11, 0, 0, 0,), 65534, dtype=torch.int64, requires_grad=False)
scale_grad_by_freq = True
mode = 0
sparse = True
per_sample_weights = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
include_last_offset = True
padding_idx = 0
torch._embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)


import torch

weight = torch.full((2, 0, 0, 6, 6,), 0, dtype=torch.float64, requires_grad=False)
indices = torch.full((2, 0, 0, 6, 6,), 2, dtype=torch.int64, requires_grad=False)
offsets = torch.full((2, 0, 0, 6, 6, 8, 6, 8, 0, 6, 0, 11, 0, 0, 0,), 65534, dtype=torch.int64, requires_grad=False)
scale_grad_by_freq = True
mode = 0
sparse = True
per_sample_weights = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
include_last_offset = True
padding_idx = 0
torch._embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)