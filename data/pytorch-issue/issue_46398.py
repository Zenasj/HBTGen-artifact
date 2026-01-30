import torch
from torch.utils.benchmark import Timer

a = torch.randn(10000, 100, 101, device='cuda')
b = torch.randn(10000, 101, 3, device='cuda')

c = torch.randn(10000, 100, 1, device='cuda')
d = torch.randn(10000, 100, 1, 3, device='cuda')

print(Timer(
    stmt='torch.einsum("bij,bjf->bif", a, b)',
    globals={'a': a, 'b': b}
).blocked_autorange())

print()

print(Timer(
    stmt='torch.einsum("bic,bicf->bif", c, d)',
    globals={'c': c, 'd': d}
).blocked_autorange())

import torch
from torch.utils.benchmark import Timer

a = torch.rand(1, 1, 16, 2, 16, 2, 16, 2, 2, 2, 2, device="cuda")
b = torch.rand(729, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, device="cuda")

print(Timer(
    stmt='(a * b).sum(dim = (-3, -2, -1))',
    globals={'a': a, 'b': b}
).blocked_autorange())

print()

print(Timer(
    stmt='torch.einsum("...ijk, ...ijk -> ...", a, b)',
    globals={'a': a, 'b': b}
).blocked_autorange())