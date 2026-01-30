import torch
from torch.autograd import Function

class Mul3ThenFlatten(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.detach().numpy() * 3
        # If I add a .contiguous() to the end of this statement, other things happen.
        return torch.tensor(z).view(-1)

    @staticmethod
    def backward(ctx, dx):
        x = ctx.saved_tensors[0]
        print("Some side effect")
        return dx.view(x.shape) * 3

x = torch.randn(2, 2, requires_grad=True)
output = Mul3ThenFlatten.apply(x)

# Modify the counter in no_grad mode.
with torch.no_grad():
    output.fill_(1)

# Doesn't print "Some side effect" :/
output.sum().backward()

x.grad  # is None, should really be something or at least an assert

with torch.no_grad():
    output.fill_(1)

with torch.no_grad():
    output.fill_(1)