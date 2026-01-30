import torch

In [2]: a=torch.tensor([20, 30, 100])

In [5]: a.logsumexp(0)
Out[5]: tensor(100.)

In [6]: torch._refs.logsumexp(a, 0)
Out[6]: tensor(inf)