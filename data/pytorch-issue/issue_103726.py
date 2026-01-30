import torch

a = torch.tensor(1., requires_grad=True)

def pack(x):
    return x

def unpack(x):
    return x

class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        intermediate = x.exp()
        ctx.save_for_backward(intermediate.clone().detach_().requires_grad_(True))
        return x.exp()

    @staticmethod
    def backward(ctx, grad_out):
        intermediate, = ctx.saved_tensors
        return grad_out * intermediate


with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out = Func.apply(a)