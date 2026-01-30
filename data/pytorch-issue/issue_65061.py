import torch

In [12]: torch.autograd.gradcheck(lambda b: torch.linalg.lstsq(a.t(), b.t()), [b], nondet_tol=1)
Out[12]: True

In [13]: torch.autograd.gradcheck(lambda b: torch.linalg.lstsq(a.t(), b.t(), driver='gelsy'), [b], nondet_tol=1)
Out[13]: True