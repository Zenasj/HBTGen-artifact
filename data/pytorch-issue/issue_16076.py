import torch

device = 'cpu'  # modify to test on a different device
N = 954  # size

# Step 1: create an conditioned singular value vector
s = torch.arange(N + 1, 1, -1, device=device).float().log()  # the condition number here is ~ log(N)
s_mat = torch.diag(s)

# Step 2: generate orthogonal matrices using QR on a random matrix
q, _ = torch.randn(N, N, device=device).qr()
assert (q @ q.t() - torch.eye(N, device=device)).abs().max() < 1e-04

# Step 3: generate the well-conditioned matrix
a = q @ s_mat @ q.t()
u, sigma, v = a.svd()
assert (a - u @ torch.diag(sigma) @ v.t()).abs().max() < 1e-04