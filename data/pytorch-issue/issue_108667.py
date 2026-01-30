import torch
t = torch.randn(10,10).to_sparse_coo().storage()