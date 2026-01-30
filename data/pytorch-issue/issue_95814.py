# %%
import torch

# %%
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4])
csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
csr.to_dense()

# %%
csr.to_sparse_coo().to_sparse_csc().to_sparse_coo().to_sparse_csr().to_dense()

# %%
ccol_indices = torch.tensor([0, 2, 4])
row_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4])
csc = torch.sparse_csc_tensor(ccol_indices, row_indices, values, dtype=torch.float64)
csc.to_dense()

# %%
csc.to_sparse_coo().to_sparse_csr().to_sparse_coo().to_dense()