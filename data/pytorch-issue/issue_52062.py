import torch
a = torch.rand(8).cuda() > 0.5
torch.flip(a[::2], [0])