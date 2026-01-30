transpose.int

transpose_

t

t_

import torch
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4], dtype=torch.float)
tensor0 = torch.sparse_csr_tensor(crow_indices, col_indices, values)
tensor0.transpose(0, 1)

import torch

crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4], dtype=torch.float)
tensor0 = torch.sparse_csr_tensor(crow_indices, col_indices, values, device=torch.device('cuda'))
tensor0.transpose(0, 1)

CSR(crow_indices, col_indices, values).transpose(0, 1) == CSR(new_crow_indices, new_col_indices, matmul(P, values))