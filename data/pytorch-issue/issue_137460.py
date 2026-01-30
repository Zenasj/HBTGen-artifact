import torch

matrix = torch.randn(3, 3)
scalar_ord = torch.tensor(2 + 3j)
result = torch._C._linalg.linalg_matrix_norm(matrix, scalar_ord, dim=(0, 1), keepdim=False)