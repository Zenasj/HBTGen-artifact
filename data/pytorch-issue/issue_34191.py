import torch
a = torch.ones(size=(5, ), device='cuda', requires_grad=True).cuda()
out, idx = torch.topk(a, 2)  
print(idx)  # e.g. tensor([4757752804209065984,                   1], device='cuda:0')