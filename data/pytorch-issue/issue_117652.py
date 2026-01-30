import torch

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, out_grad):
        x, y = ctx.saved_tensors
        return out_grad * x, out_grad * y

def f(x, y):
    return MyFunc.apply(x, y)

x = torch.ones(2, requires_grad=True)
y = torch.ones(2, requires_grad=True)
out = f(x, y)