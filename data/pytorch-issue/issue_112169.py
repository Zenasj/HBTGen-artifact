import torch
from torch._higher_order_ops.wrap import wrap

@torch.compile(backend="eager", fullgraph=True)
def f(args):
    def inner_f(args):
        x = args # assign a tuple to a local var causes graph break
        # for arg in args # is also not allowed.
        return x
    return wrap(inner_f, args)

args = ((torch.ones(1), torch.ones(1)),)
f(args)