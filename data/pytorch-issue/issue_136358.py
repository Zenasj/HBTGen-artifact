import torch

class IdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        class Node():
            pass

        a = Node()
        b = Node()
        # Induce a reference cycle
        a.b = b
        b.a = a

        s = torch.zeros(1,)
        s._attrs = {"key": "value"}
        # If the tensor is not part of ref cycle, then it is ok
        a.s = s  # Comment this line and it works fine.
        ctx.save_for_backward(s)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

t = torch.ones(1, device='cpu', requires_grad=True)

y = IdentityFunction.apply(t)

print(y.grad_fn.saved_tensors[0]._attrs)
import gc

print(gc.collect())
print(y.grad_fn.saved_tensors[0])  # This is ok
print(y.grad_fn.saved_tensors[0]._attrs)  # This fails

import torch

class IdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        class Node():
            pass

        a = Node()
        b = Node()
        # Induce a reference cycle
        a.b = b
        b.a = a

        s = torch.zeros(1,)
        s._attrs = {"key": "value"}
        # If the tensor is not part of ref cycle, then it is ok
        a.s = s  # Comment this line and it works fine.
        ctx.save_for_backward(s)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

t = torch.ones(1, device='cpu', requires_grad=True)

y = IdentityFunction.apply(t)

print(y.grad_fn.saved_tensors[0]._attrs)
import gc

print(gc.collect())
print(y.grad_fn.saved_tensors[0])  # This is ok
print(y.grad_fn.saved_tensors[0]._attrs)  # This fails