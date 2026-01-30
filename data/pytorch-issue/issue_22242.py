import torch

torch.manual_seed(0)

n = 4
L = torch.randn(n,n)
A = L.mm(L.t()).unsqueeze(0)
b = torch.randn(1, n)

A_lu_cpu = torch.lu(A)
A_lu_cuda_nopivot = torch.lu(A.cuda(), pivot=False)
A_lu_cuda_pivot = torch.lu(A.cuda(), pivot=True)
print('A_lu_cuda_nopivot\n', A_lu_cuda_nopivot)
print('-----\nA_lu_cuda_pivot\n', A_lu_cuda_nopivot)

x_cpu = b.lu_solve(*A_lu_cpu)
x_cuda_nopivot = b.cuda().lu_solve(*A_lu_cuda_nopivot)
x_cuda_nopivot_fixed = b.cuda().lu_solve(
    A_lu_cuda_nopivot[0], torch.arange(1, n+1, device='cuda:0').int())
x_cuda_pivot = b.cuda().lu_solve(*A_lu_cuda_pivot)

print(x_cpu, x_cuda_nopivot, x_cuda_nopivot_fixed, x_cuda_pivot)