import scipy.sparse as sparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(4)

def lap1d(n, dtype=np.float64):
    v  = np.ones(n, dtype=dtype)
    L1 = sparse.spdiags([-v,2*v,-v],[-1,0,1],n,n)
    return L1

n = 50000 # 40000 would be fine
A_sp = lap1d(n).tocsr()

A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float64)
# A = A.to_sparse_coo() # If converted to COO it would be fine.
x = nn.Parameter(torch.rand(n, dtype=torch.float64))

def loss(x, A):
    x = nn.functional.silu(x)
    return (A @ x).norm()

numerical = []
perturb = np.logspace(-8, 0, 50)
with torch.no_grad():
    for eps in perturb:
        L_p = loss(x + eps, A)
        L_m = loss(x - eps, A)
        numerical.append((L_p - L_m) / (2*eps))
numerical = np.array(numerical)

L = loss(x, A)
L.backward()
dLdx = x.grad.sum().item()
err = np.abs(numerical - dLdx) / np.abs(dLdx)

plt.loglog(perturb, err)
plt.xlabel('epsilon')
plt.ylabel('Relative error')
plt.grid()
plt.savefig("finite_difference.png")