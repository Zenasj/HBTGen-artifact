import torch

with torch.no_grad():
    weight = torch.randn((2, 2)).cuda()
    h, tau = torch.geqrf(weight)

h.requires_grad_(True)
tau.requires_grad_(True)

x = torch.randn((2, 2), requires_grad=True).cuda()
y = torch.ormqr(h, tau, x, left=False)
loss = y.mean()

loss.backward()