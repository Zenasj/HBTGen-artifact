import torch
a = torch.rand(2, 3, 5, requires_grad=True)
a1, a2, a3 = a.split([1,1,1], dim=1)
weight = torch.rand(2, 5)
loss = (a1 * weight + a2 * weight + a3 * weight).sum()
loss.backward()

import torch
a = torch.rand(2, 3, 5, requires_grad=True)
a1, a2, a3 = a[:, 0, :], a[:, 1, :], a[:, 2, :]
weight = torch.rand(2, 5)
loss = (a1 * weight + a2 * weight + a3 * weight).sum()
loss.backward()