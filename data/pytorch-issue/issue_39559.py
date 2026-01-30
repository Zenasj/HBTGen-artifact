import numpy as np

import torch
from torch import autograd


a = torch.rand(2, 10, requires_grad=True)

# This assumes a 2D input where you take only the idx elements of the second dimension
class MyFn(autograd.Function):
    @staticmethod
    def forward(ctx, inp, idx):
        ctx.inp_size = inp.size()
        ctx.idx = idx
        out = inp[:, :idx]
        return out

    @staticmethod
    def backward(ctx, gO):
        assert gO.is_sparse, "Only sparse gradient implemented for MyFn"
        gI = torch.sparse.FloatTensor(size=ctx.inp_size).to(gO.device, gO.dtype)
        # This is bad bad bad, but works
        # Note that you won't be able to do double backward here
        gI.indices = gO.indices
        gI.values = gO.values

        return gI, None

if False:
    # Raises the same error as your example
    b = a[:, :5]
else:
    b = MyFn.apply(a, 5)

b.backward(torch.sparse.FloatTensor(size=(2, 5)))