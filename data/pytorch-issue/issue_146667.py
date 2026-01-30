import torch
a = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32, requires_grad = True)
a_cp  = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32, requires_grad = True)
b = a.min()
b.backward()
a.grad  # Output is tensor([1., 0., 1.])
c, d = a_cp.min(dim=0)
c.backward()
a_cp.grad  # Output is tensor([0., 0., 1.])

import torch
a = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32, requires_grad = True)
a_cp  = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32, requires_grad = True)
b = a.min()
b.backward()
a.grad  # Output is tensor([0.5000, 0.0000, 0.5000])
c, d = a_cp.min(dim=0)
c.backward()
a_cp.grad  # Output is tensor([1., 0., 0.])