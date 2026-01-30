import torch
from torch.sparse import to_sparse_semi_structured

A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
A_sparse = to_sparse_semi_structured(A)