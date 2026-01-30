import torch
from resource import *


S = 256
X = torch.randn(S, S, requires_grad=True)

for i in range(10000):
    if i % 100 == 0:
        print(i, '\t', getrusage(RUSAGE_SELF).ru_maxrss, 'kB used')

    Z = X.sum(1, True) + torch.linspace(-S, S, S)
    Y = X * torch.erf(Z).sum(1, True)
    loss = Y.t().mm(Y).inverse().sum() + Y.sum()
    loss.backward()