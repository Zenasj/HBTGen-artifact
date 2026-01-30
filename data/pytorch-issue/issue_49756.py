import torch
from torch import nn
a = torch.ones(3, 1, requires_grad=True)
b = a.view_as(a)
# No grad here just to avoid issue with modifying leafs inplace
with torch.no_grad():
    a[0,0] = 3
# Here the print trigger a recompute of the grad_fn
# Removing this print makes the code work just fine
print(b)
d = torch.sum(3*b)
d.backward()