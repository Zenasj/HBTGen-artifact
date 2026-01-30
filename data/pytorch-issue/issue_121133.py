import torch.nn as nn

import torch
import torch.nn.functional as F
print(f'torch={torch.__version__}')
t = torch.arange(1.0, 4.0)
t.requires_grad = True
t_tanh = torch.tanh(t)
t_tanh.sum().backward()
M1 = 1.0 - (t_tanh)**2 ## <--  sometime ago, this generated exact match with torch tanh autograd, but now does not.

def cmp(dt, t):
  ex1 = dt == t.grad
  ex = torch.all(ex1).item()
  cnt = ex1.sum().item()
  # torch.allclose returns a bool
  app = torch.allclose(dt, t.grad)
  print(f'comparing dt vs t.grad Exact={str(ex):5s} {cnt:5}/{t.numel():5} | approx {str(app):5s}')
print(M1)
print(t.grad, "SEEMS okay but it's not!")
cmp(M1, t)