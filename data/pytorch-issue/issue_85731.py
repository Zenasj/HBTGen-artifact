import torch

xGpu = torch.tensor([-10.0, 10.0]).cuda()
pGpu = torch.poisson(xGpu)

print(pGpu)

x = torch.tensor([-10.0, 10.0])
p = torch.poisson(x)

print(p)