import torch
import torch.nn as nn


class ExampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = nn.Parameter(torch.zeros_like(x))
        print(y)
        for i in range(100):
            if y.grad is not None:
                y.grad.detach_()
                y.grad.zero_()
            loss = torch.sum(x * y)
            print(loss)
            loss.backward()
            y.data -= 0.001 * y.grad

        return y

    @staticmethod
    def backward(ctx, grad_output):
        pass

ex = ExampleFunction.apply

print(ex(torch.ones(5)))

y

requires_grad=True

loss

grad_fn

loss

grad_fn