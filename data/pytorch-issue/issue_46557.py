import torch
batch_dim = 2
A = torch.eye(3, 3, dtype=dtype, device=device)
A = A.reshape((1, 3, 3))
A = A.repeat(batch_dim, 1, 1)
A[0, -1, -1] = 0  # Now A[0] is singular
Ainv = torch.inverse(A) # Doesn't raise errors Ainv[0] contains NaNs

import torch
batch_dim = 3
A = torch.eye(3, 3, dtype=dtype, device=device)
A = A.reshape((1, 3, 3))
A = A.repeat(batch_dim, 1, 1)
A[0, -1, -1] = 0  # Now A[0] is singular
Ainv = torch.inverse(A) # Raises RuntimeError because input is singular