import torch

A = torch.tensor([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=torch.float64)
A = torch.neg(A) 
L, _ = torch.linalg.cholesky_ex(A, upper=False)

print(L)

tensor([[ -4.,   0.,   0.],
        [-12., -37.,   0.],
        [ 16.,  43., -98.]], dtype=torch.float64)

import torch

A = torch.tensor([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=torch.float64)
A = A.cuda()
A = torch.neg(A) 
L, _ = torch.linalg.cholesky_ex(A, upper=False)

print(L)

tensor([[nan, 0., 0.],
        [-0., nan, 0.],
        [0., 0., nan]], device='cuda:0', dtype=torch.float64)