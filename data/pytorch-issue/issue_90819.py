import torch

a = torch.rand(3, 3, device=torch.device('cuda')).to_sparse_csr()
a * 2