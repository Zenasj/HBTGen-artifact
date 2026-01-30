import torch

a = torch.eye(20)
a[1,1] = torch.nan
torch.linalg.lstsq(a, torch.ones(20))