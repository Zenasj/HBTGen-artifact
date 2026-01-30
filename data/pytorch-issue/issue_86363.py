import torch
from torch import zeros

def slow_inverse_th(A):
    Ainv = zeros((A.shape[0], A.shape[2], A.shape[1]))
    for i in range(A.shape[-1]):
        Ainv[i,...] = torch.linalg.pinv(A[i,...])
    return Ainv

big_A_mat = torch.randn((30, 12, 3)).double()
slow_inv_A = slow_inverse_th(big_A_mat)
fast_inv_A = torch.linalg.pinv(big_A_mat)
(slow_inv_A-fast_inv_A).abs().max(), (slow_inv_A-fast_inv_A).abs().mean()