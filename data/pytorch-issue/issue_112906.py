import torch.nn as nn

import torch

class MutatingAutogradFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, buf):
        ctx.save_for_backward(buf)
        return x

    @staticmethod
    def backward(ctx, x_grad):
        buf = ctx.saved_tensors[0]
        buf.add_(x_grad)
        return x_grad * 3, None

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.buf = torch.ones(2)

    @torch._dynamo.allow_in_graph
    def backward_mutating_fn(self, x, buf):
        return MutatingAutogradFn.apply(x, buf)

    def forward(self, x):
        tmp = self.backward_mutating_fn(x, self.buf)
        return tmp + self.buf

m = Mod()

x = torch.ones(2, requires_grad=True)
out = m(x)
# After the fw, buf should not have been mutated
print(m.buf)
out.sum().backward()
# bw has run, so buf should now be mutated
print(m.buf)
print(x.grad)