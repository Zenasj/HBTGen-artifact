import torch.nn as nn

import torch

from torch.nn.functional import linear


# Creating the customized function
class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, w):

        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(w, i)
        else:
            ctx.save_for_backward(w)
        return i.mm(w.t())

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        return grad_output.mm(saved[0]), grad_output.t().mm(saved[1]) if ctx.needs_input_grad[1] else None

# Input requiring gradient
x = torch.rand(1000, 10).cuda().requires_grad_()

# Weights NOT requiring gradient
weight = torch.rand(20, 10).cuda()


# This function performs some operations using the function f and keeps track of the
# allocated gpu memory within the function.
def test(f, x, weight):
    s = torch.cuda.memory_allocated()
    x = x + 2
    x = f(x, weight)
    print(f"Allocated memory: {torch.cuda.memory_allocated() - s}")
    return x


# I report the amount of allocated memory by torch.nn.functional
print("Test Pytorch linear")
y1 = test(linear, x, weight)

# I report the amount of allocated memory by the customized implementation
print("\nTest MyLinear")
y2 = test(MyLinear.apply, x, weight)

# Check that the obtained outputs agree
print(f"\nSame output? {y1.allclose(y2)}")

y1.norm().backward()
z1 = x.grad.clone()

x.grad.zero_()
y2.norm().backward()
z2 = x.grad.clone()

# Check that the gradient wrt input agrees.
print(f"Same gradient? {z1.allclose(z2)}")