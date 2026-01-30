x = torch.randn(3, 3, requires_grad=True).cuda()
M = torch.randn(3, 3, device="cuda").to_sparse_csr()
y = torch.mm(M, x)
g = torch.autograd.grad(y.sum(), x)

x = torch.randn(3, 3, requires_grad=True).cuda()
M = torch.randn(3, 3, device="cuda").to_sparse()
y = torch.mm(M, x)
g = torch.autograd.grad(y.sum(), x)
g

x = torch.randn(3, 3, requires_grad=True).cuda().T
M = torch.randn(3, 3, device="cuda").to_sparse_csr()
y = torch.mm(M, x)
g = torch.autograd.grad(y.sum(), x)

x = torch.randn(3, 3, requires_grad=True).cuda().T
M = torch.randn(3, 3, device="cuda").to_sparse_csr()
y = torch.mm(M, x.contiguous())
g = torch.autograd.grad(y.sum(), x)

py
import torch
x = torch.randn(3, 3)
M = torch.randn(3, 3).to_sparse_csr()
y = torch.mm(x, M) # Fails with "Could not run 'aten::zero_'"