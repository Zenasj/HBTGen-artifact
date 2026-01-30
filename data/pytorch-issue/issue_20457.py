import torch

a=torch.arange(30, dtype=torch.float).view(5,6).cuda()
ind0 = torch.arange(0,a.size(0), step=2)
gO = torch.randn(a[ind0].size()).cuda()
a.index_put_((ind0,), gO, accumulate=True)
torch.cuda.synchronize()