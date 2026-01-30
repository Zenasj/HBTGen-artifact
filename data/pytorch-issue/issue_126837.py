import torch

torch.eye(n=3, dtype=int)
# tensor([[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]])

torch.eye(n=3, dtype=float)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]], dtype=torch.float64)

torch.eye(n=3, dtype=bool)
# tensor([[True, False, False],
#         [False, True, False],
#         [False, False, True]])

torch.eye(n=3, dtype=complex) # Error

import torch

torch.eye(n=3, device='cuda:0')
torch.eye(n=3, device=0)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]], device='cuda:0')

import torch

torch.__version__ # 2.3.0+cu121