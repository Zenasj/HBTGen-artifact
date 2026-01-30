import torch

a = torch.rand(2,2, dtype=torch.cfloat)
A = a.to("mps")

b = torch.rand(N,N, dtype=torch.cfloat)
B = b.to("mps")

ab1 = torch.mm(a,b)
AB1 = torch.mm(A,B)

ab2 = torch.mm(a,torch.conj(b))
AB2 = torch.mm(A,torch.conj(B))

ab3 = torch.mm(a,torch.conj_physical(b))
AB3 = torch.mm(A,torch.conj_physical(B))

print(ab1)
print(AB1)
print(ab2)
print(AB2)
print(ab3)
print(AB3)