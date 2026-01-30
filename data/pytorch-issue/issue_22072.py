import torch
from torch.autograd import Function


class TestFunction(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        y = torch.einsum('ab,ab->a', [x, w])
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        w_grad = torch.einsum('a,ab->ab', [grad, x])
        x_grad = torch.einsum('a,ab->ab', [grad, w])
        return x_grad, w_grad


class TestFunction2(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        y = torch.einsum('ab,ab->a', [x, w])
        return y.clone()

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        w_grad = torch.einsum('a,ab->ab', [grad, x])
        x_grad = torch.einsum('a,ab->ab', [grad, w])
        return x_grad, w_grad


x = torch.ones(1, 1)
w = torch.ones(1, 1).requires_grad_()
y = TestFunction.apply(x, w)
z = y.sum()
z.backward()
print(y, z, w.grad)

x = torch.ones(1, 1)
w = torch.ones(1, 1).requires_grad_()
y = TestFunction2.apply(x, w)
z = y.sum()
z.backward()
print(y, z, w.grad)