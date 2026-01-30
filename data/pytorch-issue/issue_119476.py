import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True

class SillyCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, i):
        ctx.save_for_backward(i)
        return torch.cat([x0, x1])
    
    @staticmethod
    def backward(ctx, grad_out):
        i, = ctx.saved_tensors
        i0, i1 = i.tolist()
        g_x0, g_x1 = grad_out.split([i0, i1])
        return g_x0, g_x1, None

@torch.compile(fullgraph=True)
def f(x, i):
    i0, i1 = i.tolist()
    x0, x1 = x.split([i0, i1])
    return SillyCat.apply(x0, x1, i)

f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))