import torch
total = 0
l = []
while total < 6400:
    l.append(torch.randint(2, 10, (1,)).item())
    total += l[-1]
x = torch.randn(total, 1)
x.split(l, 0)

import functorch.dim
import torch
total = 0
l = []
while total < 6400:
    l.append(torch.randint(2, 10, (1,)).item())
    total += l[-1]
x = torch.randn(total, 1)
x.split(l, 0)