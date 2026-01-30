import random

import numpy as np
import torch

torch.manual_seed(42)

n = 2
m = 1000000

x = torch.tensor(range(1, n + 1), dtype=torch.float64, requires_grad=True)
print(x)
k = torch.tensor(0.1, dtype=torch.float64)

f = torch.mean(torch.max(torch.max((x**1).reshape([n, 1]) * torch.exp(torch.randn(n, m, dtype=torch.float64)), dim=0)[0], k))
print(f)

g = torch.autograd.grad(f, x, create_graph=True)[0]
print(g)
h1 = torch.autograd.grad(g[0], x, retain_graph=True)[0]
h2 = torch.autograd.grad(g[1], x, retain_graph=True)[0]
h = torch.stack([h1, h2])
print(h)

import numpy as np
import numdifftools as nd

np.random.seed(42)

n = 2
m = 1000000

x = np.array(range(1, n + 1))
print(x)
k = 0.1

f = lambda x: np.mean(np.maximum(np.amax((x.reshape([n, 1]) * np.exp(np.random.randn(n, m))), axis=0), k))
print(f(x))

g = nd.Gradient(f)
print(g(x))

h = nd.Hessian(f)
print(h(x))

import numpy as np
import numdifftools as nd

np.random.seed(42)

n = 2
m = 1000000

x = np.array(range(1, n + 1))
print(x)
k = 0.1

z = np.random.randn(n, m)

f = lambda x: np.mean(np.maximum(np.amax((x.reshape([n, 1]) * np.exp(z)), axis=0), k))
print(f(x))

g = nd.Gradient(f)
print(g(x))

h = nd.Hessian(f)
print(h(x))