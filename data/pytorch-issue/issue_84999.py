import torch

csr = torch.eye(3).cuda(3).to_sparse_csr()
csr_t = csr.t()

print(csr.device)
print(csr_t.device)