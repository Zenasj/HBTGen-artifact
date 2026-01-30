import time
import torch

d = 1024 * 32
A = -torch.ones(d, d)
A[0, 0] = 111
A[10, 10] = 222     # most entries in A are < 0

T = torch.relu(A)         # materializes very sparse T as dense first
S = T.to_sparse_csr()     # would be nice to have something like S = torch.sparse_relu(A)
                          # but that is not the point of this bug yet