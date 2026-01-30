py
import torch
from torch.func import grad

class Cube(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x ** 3, x

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        cube, x = outputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output, grad_x):
        # NB: grad_x intentionally not used in computation
        x, = ctx.saved_tensors
        result = grad_output * 3 * x ** 2
        return result

def f(x):
    cube, inp = Cube.apply(x)
    return cube

x = torch.tensor(1., requires_grad=True)

print("============= functorch")
grad_x = grad(f)(x)
grad_out = torch.ones_like(grad_x)
result = torch.autograd.grad(grad_x, x, grad_out)
print(result)

print("============= pytorch")
y = f(x)
grad_out = torch.ones_like(y)
grad_x2, = torch.autograd.grad(y, x, create_graph=True)
expected = torch.autograd.grad(grad_x2, x, grad_out)
print(expected)