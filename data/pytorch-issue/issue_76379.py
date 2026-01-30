import torch
crow_indices = [0, 2, 4]
col_indices = [0, 1, 0, 1]
values = [1, 2, 3, 4]
t = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64), torch.tensor(col_indices, dtype=torch.int64), torch.tensor(values), dtype=torch.double)
t.to(dtype=torch.float)