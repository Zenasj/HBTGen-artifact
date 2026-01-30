import torch

def my_square(x):
    return x**2

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.detach().requires_grad_()
        ctx.x = x

        with torch.enable_grad():
            ctx.result = my_square(x)
        return ctx.result.detach()

    @staticmethod
    def backward(ctx, grad_out):
        (grad_x,) = torch.autograd.grad(
            ctx.result,
            ctx.x,
            grad_outputs=grad_out,
            create_graph=True,
        )

        return grad_x

x = torch.tensor([2, 3, 4], requires_grad=True, dtype=torch.double)
assert torch.autograd.gradcheck(Square.apply, x)
assert torch.autograd.gradgradcheck(Square.apply, x)

import torch

def my_square(x):
    return x**2

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.enable_grad():
          result = my_square(x)
        ctx.save_for_backward(x, result)

        return result.clone()

    @staticmethod
    def backward(ctx, grad_out):
        x, result = ctx.saved_tensors
        (grad_x,) = torch.autograd.grad(
            result,
            x,
            grad_outputs=grad_out,
            create_graph=True,
        )

        return grad_x

x = torch.tensor([2, 3, 4], requires_grad=True, dtype=torch.double)
out = Square.apply(x)
assert torch.autograd.gradcheck(Square.apply, x)
assert torch.autograd.gradgradcheck(Square.apply, x)