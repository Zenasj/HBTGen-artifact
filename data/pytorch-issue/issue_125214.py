import torch.nn as nn

import torch

shapes = [(19, 1024), (18, 1024), (22, 1024), (19, 1024), (13, 1024),
          (18, 1024), (21, 1024), (17, 1024), (22, 1024), (19, 1024)]

a = torch.nested.as_nested_tensor(
    [torch.randn(*shape, device="cuda") for shape in shapes],
    layout=torch.jagged
)

print(a.shape, a.dim())  # torch.Size([10, j1, 1024]) 3

# do projection
lin = torch.nn.Linear(1024, 1024, bias=False, device="cuda")
q = lin(a)

print(q.shape, q.dim())  # torch.Size([10, j1, 1024]) 3

# split heads
p = q.unflatten(-1, [8, 128])
# alternative reshape() calls:
# p = q.reshape(-1, -1, 8, 128)
# p = q.reshape(10, -1, 8, 128)

print(p.shape, p.dim())  # torch.Size([10, j1, 8, 128]) 4

from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ...

# njt1: shape (B, R1, D)
# njt2: shape (B, R2, D)
# output: shape (B, R1, D) x (B, D, R2) -> (B, R1, R2); two ragged dims
output = matmul(njt1, njt2.transpose())