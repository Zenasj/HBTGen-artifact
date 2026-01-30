import torch

b = torch.rand([4,3])
c = torch.rand([3,4])
a = torch.mm(b, c)
_a = torch.mm(b, c).cuda()

torch.topk(a, k=a.size(0), sorted=False)[1]
torch.topk(_a, k=_a.size(0), sorted=False)[1]