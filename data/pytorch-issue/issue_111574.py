import torch

torch.sparse.mm(src.to_sparse_csc(), other) / deg.view(-1, 1).clamp_(min=1)

torch.sparse.mm(src.to_sparse_csc().to_sparse_csr(), other) / deg.view(-1, 1).clamp_(min=1)