import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)  # Dummy input
v = torch.vander(x, N=4, increasing=True)
loss = v.sum()
loss.backward()

import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)  # Dummy input
v = torch.stack([x**0, x, x**2, x**3], dim=1)
loss = v.sum()
loss.backward()