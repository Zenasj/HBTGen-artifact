import torch

i = torch.LongTensor(idx)
values = torch.FloatTensor([1] * len(idx))
M = torch.sparse.FloatTensor(i.t(), values, torch.Size([4847571, 4847571]))
N = M.shape[1]
v = torch.rand(N, 1).float()
values = torch.FloatTensor([(1 - d)/N] * len(indices))
temp = torch.sparse.FloatTensor(i.t(), values, torch.Size([4847571,
                                    4847571]))
if torch.cuda.is_available():
     v = v.cuda()
     M = M.cuda()
     temp = temp.cuda()

v = v / torch.norm(v, 1)
M_hat = self.d * M + temp
for i in range(num_iter):
     v = torch.mm(M_hat, v)