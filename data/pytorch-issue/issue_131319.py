import torch
tensor1 = torch.rand([3, 3, 5, 5], dtype=torch.float64).to_sparse_csc()
tensor2 = torch.rand([3, 3, 5, 5], dtype=torch.float64).to_sparse_csc()
res = torch.triangular_solve(tensor1, tensor2, upper=False)