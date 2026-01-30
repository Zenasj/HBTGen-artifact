import torch
u1 = torch.unique(torch.rand(3,3), sorted=False)
u2 = torch.unique((10 * torch.rand(3,3)).round().to(dtype=torch.int), sorted=False)
u3 = torch.unique(torch.rand(3,3).round().to(dtype=torch.bool), sorted=False)