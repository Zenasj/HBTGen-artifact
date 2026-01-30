import torch


class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, c):
        return a.svd()

    @staticmethod
    def backward(ctx, a, b, c):
        return a.svd()


op = MyOp.apply
a = torch.randn(5, 5).requires_grad_(True)
op(a, a[0], a)[0].sum().backward()