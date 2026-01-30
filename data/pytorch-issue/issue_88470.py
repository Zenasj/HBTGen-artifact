py
import torch

a = torch.arange(4.0)

not_zero = 0.001

b = torch.where(a != 0, a, not_zero)
c = a.where(a != 0, not_zero)  # Error!

assert b.equal(c)

py
import torch

a = torch.arange(4.0)

not_zero = 0.001

b = torch.where(a != 0, a, not_zero)
c = a.where(a != 0, torch.full_like(a, not_zero))  # manually create a tensor for the replacement value

assert b.equal(c)

py
c = a.where(a != 0, torch.broadcast_to(torch.tensor(not_zero), a.shape))