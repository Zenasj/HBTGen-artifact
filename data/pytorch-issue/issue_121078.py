import torch

LU_data = torch.randn(1, 1, 1, 1, 1)
LU_pivots = torch.tensor([0], dtype=torch.int32).contiguous()
P, L, U = torch.lu_unpack(LU_data, LU_pivots)

print(P)

import torch

LU_data = torch.randn(1, 1, 1, 1, 1)
LU_pivots = torch.tensor([0], dtype=torch.int32).contiguous()
LU_data = LU_data.cuda()
LU_pivots = LU_pivots.cuda()
P, L, U = torch.lu_unpack(LU_data, LU_pivots)

print(P)

tensor([[[[[1.]]]]], device='cuda:0')