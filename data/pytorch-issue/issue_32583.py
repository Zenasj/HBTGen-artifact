import torch
t = torch.zeros(10, dtype=torch.bool).cuda()
t.roll(1)