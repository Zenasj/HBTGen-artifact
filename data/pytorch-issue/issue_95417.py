import torch

device = torch.device('mps')

t = torch.zeros(5, device=device)
u = torch.ones(2, device=device).long()
t[2:4] = u
print(t)

tensor([1., 1., 0., 0., 0.], device='mps:0')

tensor([0., 0., 1., 1., 0.], device='mps:0')