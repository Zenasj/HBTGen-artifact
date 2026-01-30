import torch

t = torch.zeros(15,20).cuda()
M = torch.zeros(10,5).cuda()
print(t.shape,M.shape)
t.matmul(M)
print(t.matmul(M).shape)