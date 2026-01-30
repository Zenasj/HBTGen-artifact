import torch

In [7]: a=torch.ones([1838860800], dtype=torch.float, device="cuda:1")

In [8]: a.mean()
Out[8]: tensor(1., device='cuda:1')