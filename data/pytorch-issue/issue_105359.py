import torch

factor_matrix = torch.load("rank7_idx0.1.3.0_iter100_factor.pt")
factor_matrix = factor_matrix.to("cuda:0")
torch.linalg.eigh(factor_matrix)  # will error with "failed to compute eigendecomposition"

print(factor_matrix)  # illegal memory access