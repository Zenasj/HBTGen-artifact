import torch

ret = a.max(dim=0)
ret1 = torch.max(a, dim=0, out=ret)