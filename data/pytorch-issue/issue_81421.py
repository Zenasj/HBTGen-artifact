import torch
a = torch.randn(40, 50).relu()
a_csr = a.to_sparse_csr().requires_grad_()
torch.mm(a_csr, a_csr.transpose(0, 1)).sum().backward()
print(a_csr.grad)