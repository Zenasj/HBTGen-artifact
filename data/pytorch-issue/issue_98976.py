import torch
A = torch.eye(9).to_sparse_coo()
B = torch.zeros_like(A)
A.detach().copy_(B)
print(A)