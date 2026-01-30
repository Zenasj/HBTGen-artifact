import torch

h, w, d = 800, 1216, 12
n = 133
A = torch.randn(n, d).cuda()
B = torch.randn(h, w, d).cuda()
A.requires_grad = True
B.requires_grad = True

B = B.reshape(-1, d).contiguous()
dist = torch.cdist(A, B)
loss = dist.sum()
loss.backward()