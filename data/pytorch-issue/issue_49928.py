import torch

for i in range(510, 520):
    a = torch.zeros((4, i, 32)).normal_(0, 1).cuda()
    b = torch.zeros((4, i, 32)).normal_(0, 1).cuda()
    a.requires_grad = True
    b.requires_grad = True
    c = torch.cdist(a, b, p=1.0)
    print(c.size())
    d = c.mean()
    d.backward()
    print(a.grad.size(), b.grad.size())