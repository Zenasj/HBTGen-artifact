import torch

c = torch.zeros(3, 3)
c *= float('nan')
try:
    torch.linalg.eigh(c)
except RuntimeError as e:
    print(e)
# torch.linalg.eigh: the algorithm failed to converge; 2 off-diagonal elements of an intermediate tridiagonal form did not converge to zero.

try:
    torch.linalg.eigh(c.unsqueeze(0))
except RuntimeError as e:
    print(e)
# torch.linalg.eigh: For batch 0: U(2,2) is zero, singular U.