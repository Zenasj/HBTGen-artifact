import torch
from numpy.testing import assert_allclose

lineSize = 256 # change this to 128, assert will pass

def _kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    return AB

A = torch.randint(2, [lineSize, lineSize]).int()
B = torch.randint(2, [lineSize, lineSize]).int()

AB = _kronecker(A, B)
ABT = _kronecker(A.t(), B.t())

assert_allclose(AB.t().detach().numpy(), ABT.detach().numpy())

import torch

# SIZE = 128
SIZE = 256

def kronecker(A, B):
    X = A.reshape(A.shape[0], 1, A.shape[1], 1)
    Y = B.reshape(1, B.shape[0], 1, B.shape[1])
    return X.mul(Y).reshape(A.size(0) ** 2, A.size(0) ** 2)
    # return X.add(Y).reshape(A.size(0) ** 2, A.size(0) ** 2)

A = torch.randint(2, (SIZE, SIZE))
B = torch.randint(2, (SIZE, SIZE))

AB = kronecker(A, B)
AB_T = kronecker(A.T, B.T)
# AB_T = kronecker(A.T.contiguous(), B.T.contiguous())

assert AB.T.allclose(AB_T)