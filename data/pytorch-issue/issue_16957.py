import torch
a = torch.randn(5, 5)
a_, tau = a.geqrf()

b = torch.empty_like(a)
c = torch.empty_like(tau)
torch.geqrf(a, out=(b, c))