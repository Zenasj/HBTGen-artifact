import torch.nn as nn

import torch as th

th.autograd.set_detect_anomaly(True)

x = th.nn.Parameter(th.randn(3))
y = th.nn.Parameter(th.randn(3))
z = th.nn.Parameter(th.randn(3))

optim_a = th.optim.Adam([x, y], 1e-3)
optim_b = th.optim.Adam([y, z], 1e-3)

loss_a = ((x+y)**2).sum()

loss_b = (y**2).sum()
#loss_b = ((y+z)**2).sum() # Oddly, if you compute this instead, there is no inplace error.

loss_a.backward(retain_graph=True)
optim_a.step()

# If you move the computation of `loss_b` down here instead, there is also no inplace error.
loss_b.backward()