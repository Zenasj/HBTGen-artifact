import torch

class Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_0, x_1, fn_0, fn_1):
        ctx.x_0, ctx.x_1 = x_0, x_1
        ctx.y_0, ctx.y_1 = fn_0(x_0), fn_1(x_1)
        return ctx.y_0, ctx.y_1

    @staticmethod
    def backward(ctx: Any, dy_0, dy_1):
        dx_0 = torch.autograd.grad(ctx.y_0, ctx.x_0, grad_outputs=dy_0) # Autograd falls into Infinitive Recusive and ends with stack overflow
        dx_1 = torch.autograd.grad(ctx.y_1, ctx.x_1, grad_outputs=dy_1) # Autograd falls into Infinitive Recusive and ends with stack overflow
        return (dx_0, dx_1)

class Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_0, x_1, fn_0, fn_1):
        x_0 = x_0.detach().requires_grad_()
        x_1 = x_1.detach().requires_grad_()
        ctx.x_0, ctx.x_1 = x_0, x_1
        with torch.enable_grad():
            ctx.y_0, ctx.y_1 = fn_0(x_0), fn_1(x_1)
        return ctx.y_0.detach(), ctx.y_1.detach()

    @staticmethod
    def backward(ctx, dy_0, dy_1):
        dx_0, = torch.autograd.grad(ctx.y_0, ctx.x_0, grad_outputs=dy_0) # Autograd falls into Infinitive Recusive and ends with stack overflow
        dx_1, = torch.autograd.grad(ctx.y_1, ctx.x_1, grad_outputs=dy_1) # Autograd falls into Infinitive Recusive and ends with stack overflow
        return dx_0, dx_1, None, None

inp = (torch.rand(2, requires_grad=True), torch.rand(2, requires_grad=True))

Merge.apply(*inp, torch.exp, torch.exp)[0].sum().backward()