import torch
from torch.autograd import Function

class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x 

    @staticmethod
    def backward(ctx, g):
        return g 

x = torch.zeros(1, requires_grad=True)
y = MyFunction.apply(x)
y.backward()
print(y.grad_fn.metadata)
g = y.grad_fn
del y 
print(g.metadata)