import torch

print(torch.__version__)

sym_0 = (1, 3, 8, 3)
sym_1 = torch.strided
sym_2 = 'cuda'
sym_3 = (1, 48)

v0 = torch.randn(size=sym_0, dtype=None, layout=sym_1, device=sym_2)
v1 = torch.rand(size=sym_3, device=sym_2)
torch.ops.aten._adaptive_avg_pool2d_backward(grad_output=v0, self=v1)

import torch

print(torch.__version__)

sym_0 = (1, 3, 8, 3)
sym_1 = torch.strided
sym_2 = 'cpu'
sym_3 = (1, 48)

v0 = torch.randn(size=sym_0, dtype=None, layout=sym_1, device=sym_2)
v1 = torch.rand(size=sym_3, device=sym_2)
torch.ops.aten._adaptive_avg_pool2d_backward(grad_output=v0, self=v1)