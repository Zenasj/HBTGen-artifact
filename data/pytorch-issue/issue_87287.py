import torch

torch.use_deterministic_algorithms(True)

a = torch.zeros(1, 10, 16).cuda()
mask = torch.ones(1, 10, dtype=torch.bool).cuda()
b = torch.zeros(16).cuda()
a[mask] = b