import torch

In [133]: t
Out[133]: tensor([0, 1, 2, 3, 4, 5])

In [134]: torch.diff(t, n=-3)
Out[134]: tensor([0, 1, 2, 3, 4, 5])

In [135]: torch.diff(t, n=0)
Out[135]: tensor([0, 1, 2, 3, 4, 5])

In [139]: torch.diff(t, n=-1) is t
Out[139]: True