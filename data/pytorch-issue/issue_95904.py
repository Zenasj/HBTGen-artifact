import torch

class TestFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        b = a + 1
        c = b.view(-1)
        ctx.save_for_backward(c)
        return b

    @staticmethod
    def backward(ctx, *flat_args):
        raise RuntimeError("")

a = torch.ones(64656, 640, device='cuda', requires_grad=True)
import gc
gc.collect()
mem_before = torch.cuda.memory_allocated()
TestFn.apply(a)
gc.collect()
mem_after = torch.cuda.memory_allocated()

import torch

class CompiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = a + a
        d = c.view(-1)
        e = b * d
        ctx.save_for_backward(b, d)
        return c, e
        # No segfault if I return a list instead
        # return [c, f]

    @staticmethod
    def backward(ctx, *deduped_flat_tensor_args):
        raise RuntimeError()

inps = [
    torch.ones(2, 2, device='cuda', requires_grad=True),
    torch.ones(4, device='cuda', requires_grad=True),
]
import gc
gc.collect()
mem_before = torch.cuda.memory_allocated()
import pdb; pdb.set_trace()
out = CompiledFunction.apply(*inps)
del out
gc.collect()
mem_after = torch.cuda.memory_allocated()
print("inps[0]")
# printing inps[1] segfaults, since we apparently deallocated it.
print(inps[1])
print("inps[1]")
print(inps[1])
print("DONE")