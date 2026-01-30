import torch
t = torch.tensor([[1, 2+3j], [2+3j, 1j]])

print(t@t.conj_physical())
#tensor([[14.+0.j,  5.-5.j],
#        [ 5.+5.j, 14.+0.j]])

print(t@t.conj())
#tensor([[14.+0.j,  5.-5.j],
#        [ 5.+5.j, 14.+0.j]])

print(t@t.t().conj_physical())
#tensor([[14.+0.j,  5.-5.j],
#        [ 5.+5.j, 14.+0.j]])

print(t@t.t().conj())
#tensor([[-4.+12.j, -1.+5.j],
#        [-1.+5.j, -6.+12.j]])

import numpy as np

print(torch.cov(t))
#tensor([[-3.+3.5000j,  1.-4.0000j],
#        [ 3.-3.5000j, -1.+4.0000j]])

print(np.cov(t.numpy()))
#[[ 5.+0.j -4.-2.j]
# [-4.+2.j  4.+0.j]]

tensor([[14.+0.j,  5.-5.j],
        [ 5.+5.j, 14.+0.j]])
tensor([[14.+0.j,  5.-5.j],
        [ 5.+5.j, 14.+0.j]])
tensor([[14.+0.j,  5.-5.j],
        [ 5.+5.j, 14.+0.j]])
tensor([[14.+0.j,  5.-5.j],
        [ 5.+5.j, 14.+0.j]])