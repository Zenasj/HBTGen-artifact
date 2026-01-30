import torch

A=torch.ones(1,2,1, device="cuda", dtype=torch.complex128)
B=torch.ones(1,3,1, device="cuda", dtype=torch.complex128).transpose(1,2).conj()
print(torch.bmm(A,B))

tensor([[[1.+0.j, 0.+0.j, 0.+0.j],
         [1.+0.j, 0.+0.j, 0.+0.j]]], device='cuda:0', dtype=torch.complex128)