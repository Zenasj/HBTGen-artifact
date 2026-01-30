import torch

idx = torch.tensor([0, 1])
b = torch.zeros(5).cuda()
c = torch.tensor([1., 2. ])  # should be c = torch.tensor([1., 2. ]).cuda()
b.index_put_((idx, ), c, accumulate=True)