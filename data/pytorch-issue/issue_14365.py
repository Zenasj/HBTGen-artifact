import torch
from torch.autograd import Function

class Flatten3(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.view(-1), y.view(-1)

    @staticmethod
    def backward(ctx, dx, dy):
        x, y = ctx.saved_tensors
        return dx.view(x.shape), dy.view(y.shape)

inputs = [torch.randn(2, 2, requires_grad=True) for _ in range(2)]

# a, b are all DifferentiableViewImpl with output_nr = {0, 1} respectively
a, b = Flatten3.apply(*inputs)

# Modify the counter in no_grad mode.
# b is still a DifferentiableViewImpl with output_nr = 1 (no_grad prevents
# a new autograd function from being created)
with torch.no_grad():
    b.zero_()

# Throws assert because:
# 1) b has been modified in place
# 2) b is a view
# 3) b's output_nr is not 0.
z = b + a