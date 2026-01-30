import torch

# Create a low-rank complex matrix.
U = torch.complex(torch.randn(50,3), torch.randn(50,3))
V = torch.complex(torch.randn(3,50), torch.randn(3,50))
S = torch.randn(3, dtype=torch.cfloat)
A = torch.matmul(U, torch.matmul(torch.diag(S), V))

# works
U1, S1, V1 = torch.linalg.svd(A, full_matrices=False)

# Crashes.
U2, S2, V2 = torch.svd_lowrank(A, q=5)
# RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::complex<float> != float