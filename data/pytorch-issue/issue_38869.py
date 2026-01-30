import torch as th

t1 = th.tensor([[1, 2], [3,4]]).float().requires_grad_()
t2 = th.argmin(t1, dim=1)
t3 = t1[t2]
t3.sum().backward()

import torch as th

t1 = th.tensor([[1, 2], [3,4]]).float().requires_grad_()
t2 = th.argmin(t1, dim=1)
t3 = t1[t2.detach()]
t3.sum().backward()