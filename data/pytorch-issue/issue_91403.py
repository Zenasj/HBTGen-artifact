py
import torch
from functorch import vmap
x = torch.randn(32, 3)
y = vmap(torch.triu)(x)

results = []
for xi in x:
  y = torch.triu(xi)
  results.append(y)