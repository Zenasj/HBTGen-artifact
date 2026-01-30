import torch

values = torch.tensor([1., 2., 3., 1., 2., 3., 4., 1., 2.])
crow_indices = [0, 3, 7, 9] # Python list instead of torch Tensor!
col_indices = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1])
size = (3, 4)

torch.sparse_csr_tensor(crow_indices,
                        col_indices,
                        values,
                        size)