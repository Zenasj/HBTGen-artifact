import torch

LU_data = torch.empty((0, 3, 8, 0, 6,), dtype=torch.float64)
LU_pivots = torch.empty((3,), dtype=torch.int32)
unpack_data = True
unpack_pivots = True
torch.lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots)