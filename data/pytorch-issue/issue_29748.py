import torch

In [9]: torch.bernoulli(torch.ones(3))
Out[9]: tensor([1., 1., 1.])

In [10]: torch.bernoulli(torch.ones(3), generator=None)
Out[10]: tensor([1., 1., 1.])