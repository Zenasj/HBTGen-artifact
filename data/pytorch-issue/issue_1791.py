import torch

Pidx = type(LU_data)(range(sz)).repeat(nBatch, 1).long()

for i in range(sz):
    k = LU_pivots[:, i] - 1
    t = Pidx[:, i].clone()
    Pidx[:, i] = torch.gather(Pidx, 1, k.unsqueeze(1).long())
    Pidx.scatter_(1, k.unsqueeze(1).long(), t.unsqueeze(1))

P = type(LU_data)(nBatch, sz, sz).zero_()
for i in range(nBatch):
    P[i].scatter_(0, Pidx[i].unsqueeze(0), 1.0)

def apply(func, M):
    tList = [func(m) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)
    return res 

apply(torch.triu,  qVar)

B, N = 4, 10
qVar = torch.triu(torch.Tensor(N, N)).expand(B, N, N)