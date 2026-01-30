import torch

i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
s = torch.sparse.FloatTensor(i, v, torch.Size([2,3]))

d = torch.rand(3, 4)
d2 = torch.rand(2, 4)

s.requires_grad_()

res = torch._sparse_addmm(d2, s, d)

res.sum().backward()