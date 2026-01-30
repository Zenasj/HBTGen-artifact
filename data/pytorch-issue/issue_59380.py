import torch

x=torch.randn(2, 3, dtype=torch.cdouble, device='cuda', requires_grad=True).conj()
y=torch.randn(3, 2, dtype=torch.cdouble, device='cuda',requires_grad=True).transpose(0, 1).conj()
z=torch.randn(3, 3, dtype=torch.cdouble, device='cuda',requires_grad=True).transpose(0, 1).conj()

out=torch.addmm(x, y, z)

out.sum().backward()

print(torch.cuda.memory_summary())

M = torch.randn(k, m, dtype=dtype, device=device).transpose(0, 1)
mat1 = torch.randn(m, n, dtype=dtype, device=device)
mat2 = torch.randn(k, n, dtype=dtype, device=device).transpose(0, 1)

task1 = "torch.addmm(M.conj_physical(), mat1.conj_physical(), mat2.conj_physical())"
task2 = "torch.addmm(M.conj(), mat1.conj(), mat2.conj())"