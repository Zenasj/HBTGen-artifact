import torch.nn as nn

import torch

t = torch.tensor([-1.0, 0, 1.0], dtype = torch.float32)
t.requires_grad = True
t_htnh = torch.nn.functional.hardtanh(t)
t_htnh.backward(torch.ones_like(t_htnh))

print(t.grad)

# yields tensor([0., 1., 0.]), but should be tensor([1., 1., 1.])