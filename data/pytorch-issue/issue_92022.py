import torch

csr = torch.sparse_csr_tensor((0, 1, 2), (0, 1), (1, 1), dtype=torch.float32, requires_grad=True)
csr2 = csr.to_sparse(layout=torch.sparse_csr)
x = torch.ones((2, 1), dtype=torch.float32)
y = torch.matmul(csr2, x)
z = torch.sum(y)
z.backward()
print(csr.grad)

csr = torch.sparse_csr_tensor((0, 1, 2), (0, 1), (1, 1), dtype=torch.float32, requires_grad=True)
csr2 = csr.to_sparse(layout=torch.sparse_csr).detach().requires_grad_(True)
x = torch.ones((2, 1), dtype=torch.float32)
y = torch.matmul(csr2, x)
z = torch.sum(y)
z.backward()
print(csr2.grad)   # UPDATED