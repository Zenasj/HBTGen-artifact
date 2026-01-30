import torch.nn.functional as F

import torch
from torch.nn import functional as F
t = torch.tensor([[
          [[ 0.0000, -0.4945],
          [ 0.0000,  0.3474],
          [ 0.0000,  0.8383],
          [ 0.0000,  0.5882],
          [ 0.0000,  0.3417]],
]])
t = t.to('cpu')
for _ in range(int(1e7)):  # sometimes you need more runs 
    bins = F.gumbel_softmax(t, tau=0.15, hard=False, dim=-1)
    assert not bins.isnan().any().item(), f"Found {bins.isnan().sum()} nans"

t = t.to('cuda:0')
for _ in range(int(1e7)):  # no bug here
    bins = F.gumbel_softmax(t, tau=0.15, hard=False, dim=-1)
    assert not bins.isnan().any().item(), f"Found {bins.isnan().sum()} nans"