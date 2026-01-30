import torch
rng = torch.Generator(device="cuda")
idx_pts = torch.randperm(n=50, generator=rng, device="cuda")

import torch
rng = torch.Generator(device="cuda:0")
idx_pts = torch.randperm(n=50, generator=rng, device="cuda")