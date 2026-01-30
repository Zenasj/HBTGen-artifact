import torch

g = torch.randn(10, dtype=torch.complex64, device='cuda')
g.requires_grad = True
opt = torch.optim.SGD([g], lr=1)

h = torch.ones(g.shape, device=g.device, dtype=g.dtype)
h[:1] = torch.abs(g[:1]).mean().to(dtype=g.dtype)

loss = h.mean()
loss.backward()