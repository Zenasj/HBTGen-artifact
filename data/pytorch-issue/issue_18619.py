import torch


def mult1(x):
    return x.prod(dim=-1).prod(dim=-1)


class Mult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = mult1(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return (grad_output * y)[:, None, None] / x


mult2 = Mult.apply


def check_gradgrad_repeated(x, y):
    gy, = torch.autograd.grad(y[0], x, create_graph=True)
    ggy_1, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
    gy, = torch.autograd.grad(y[0], x, create_graph=True)
    ggy_2, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
    print(ggy_1[0, 0, 1], ggy_2[0, 0, 1])


x = torch.ones(2, 4, 4).requires_grad_()
check_gradgrad_repeated(x, mult1(x))
check_gradgrad_repeated(x, mult2(x))

import torch


class Double(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x ** 2
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors
        return grad_output * 2 * x


# this is equivalent, but uses the output of .forward() in .backward()
class Double2(Double):
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * 2 * y / x


double = Double.apply
double2 = Double2.apply

x = torch.tensor(2).double().requires_grad_()

print(
    'gradcheck',
    torch.autograd.gradcheck(double, x),
    torch.autograd.gradgradcheck(double, x),
    torch.autograd.gradcheck(double2, x),
    torch.autograd.gradgradcheck(double2, x),
)

y = double(x)
torch.autograd.grad(y, x, create_graph=True)
torch.autograd.grad(y, x)
print('double() ok')

y = double2(x)
torch.autograd.grad(y, x, create_graph=True)
torch.autograd.grad(y, x)
print('double2() ok')

import torch


class Double(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x + 1
        ctx.save_for_backward(x, y)
        return y ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors
        return grad_output * 2 * (x + 1)


class Double2(Double):
    @staticmethod
    def backward(ctx, grad_output):
        _, y = ctx.saved_tensors
        return grad_output * 2 * y


double = Double.apply
double2 = Double2.apply

x = torch.tensor(2).double().requires_grad_()

print('gradcheck double', torch.autograd.gradcheck(double, x))
print('gradgradcheck double', torch.autograd.gradgradcheck(double, x))
print('gradcheck double2', torch.autograd.gradcheck(double2, x))
print('gradgradcheck double2', torch.autograd.gradgradcheck(double2, x))