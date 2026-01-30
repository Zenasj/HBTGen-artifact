py
import torch
import torch._dynamo

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.x_dim = x.dim
        return x.sin()
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * x.cos() * ctx.x_dim

@torch.compile(backend='aot_eager', fullgraph=True)
def f(x):
    return Foo.apply(x)

x = torch.randn([1], requires_grad=True)
f(x)