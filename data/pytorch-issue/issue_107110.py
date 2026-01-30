import torch

dst = torch.full([7, 2], -1).int().cuda()
dst_with_bizarre_stride = torch.as_strided(dst, [7, 2], [1,2])

src = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]).int().cuda()
src_t = src.t()

dst_with_bizarre_stride.copy_(src_t)