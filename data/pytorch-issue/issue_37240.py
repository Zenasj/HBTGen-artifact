import torch
from torch import autograd
autograd.set_detect_anomaly(True)

import warnings
warnings.simplefilter("always")

return_nan = True

class MyBadFn(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp ** 2

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        res = 2 * grad_out * inp
        if return_nan:
            res = grad_out / 0 * 0
        return res


inp = torch.rand(10, requires_grad=True)

out = MyBadFn.apply(inp)

out.sum().backward()