import torch

a = torch.as_tensor([-0.2, 1.0, 2.3])
b = a + 1
idx = torch.nonzero(a> 0).view(-1)
b.scatter_(0, idx, 0)