import torch

a = torch.randn(2, 3).requires_grad_(False).to_sparse().requires_grad_(True)
a
tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
[0, 1, 2, 0, 1, 2]]),
values=tensor([-1.6929, -0.6308, 1.0252, 1.2527, 0.9184, 1.7817]),
size=(2, 3), nnz=6, layout=torch.sparse_coo, requires_grad=True)
a = torch.randn(2, 3).requires_grad_(True).to_sparse().requires_grad_(True)
a
tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
[0, 1, 2, 0, 1, 2]]),
values=tensor([ 0.3564, 1.5703, 1.2693, -0.4577, 0.7191, -1.0451]),
size=(2, 3), nnz=6, layout=torch.sparse_coo, grad_fn=NotImplemented)