import torch.nn as nn

import torch
from torch import nn


class BackwardBreaks(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        assert len(args) == 3
        model = args[0]
        inp1, inp2 = args[1], args[2]
        ctx.model = model
        ctx.save_for_backward(inp1, inp2)
        return model.f(inp1) + model.g(inp2)

    @staticmethod
    def backward(ctx, *grad_outputs):
        model, inps = ctx.model, ctx.saved_tensors
        inp1, inp2 = inps
        grad_outputs, = grad_outputs
        with torch.enable_grad():
            print(torch._C.is_grad_enabled())
            grad1, = torch.autograd.grad(
                inp1 * 3, inp1, grad_outputs=grad_outputs
            )
            print(torch._C.is_grad_enabled())
            grad2, = torch.autograd.grad(
                inp2 ** 2, inp2, grad_outputs=grad_outputs
            )
        return None, grad1, grad2


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def f(self, inp):
        return inp * 2.

    def g(self, inp):
        return inp ** 2.


def backward_breaks_wrapper(model, inp1, inp2):
    return BackwardBreaks.apply(model, inp1, inp2)


def main():
    model = Model()
    inp1 = torch.randn(3, 1, requires_grad=True)
    inp2 = torch.randn(3, 1, requires_grad=True)

    result = backward_breaks_wrapper(model, inp1, inp2)
    result.backward(torch.ones(3, 1))
    print(inp1.grad)
    print(inp2.grad)


if __name__ == '__main__':
    main()