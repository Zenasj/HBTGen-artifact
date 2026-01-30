import torch

torch.manual_seed(0)
n = 4
L = torch.randn(n,n)
A = L.mm(L.t())
b = torch.randn(n)
b.lu_solve(*torch.lu(A))

b.unsqueeze(0).lu_solve(*torch.lu(A.unsqueeze(0)))