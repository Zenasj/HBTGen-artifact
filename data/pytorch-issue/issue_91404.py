py
import torch
from functorch import vmap
x = torch.randn(32, 3, 3, 3)
y = vmap(torch.trace)(x)

results = []
for xi in x:
  y = torch.trace(xi)
  results.append(y)