import torch

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        intermediate = x + 1
        return intermediate.view(-1)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError("UNUSED")


a = torch.ones(2, requires_grad=True)
out = MyFunc.apply(a)
out.mul_(2)

def f(x):
    intermediate = x + 1
    return intermediate, intermediate.view(-1)