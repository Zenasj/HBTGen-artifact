torch.distributed.all_reduce

autograd.Function

torch.distributed._functional_collectives.all_reduce_inplace

copy_

import torch
torch.set_default_device('cuda')

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad1, grad2):
        return grad1.copy_(grad2)

@torch.compile(backend="aot_eager", fullgraph=True)
def f(x):
    return Foo.apply(x)

x = torch.randn(3, requires_grad=True)
result = f(x)
print(result)

torch.distributed