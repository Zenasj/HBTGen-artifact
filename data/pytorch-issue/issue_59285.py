import torch

t = torch.rand(2, 2)
csr = t.to_sparse_csr().clone()