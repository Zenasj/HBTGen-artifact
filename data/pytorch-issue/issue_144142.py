import torch
from torch.autograd import Function

class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x, [1, 2, 3]  # Tensor and list of integers

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        return grad_output1

x = torch.tensor(2.0, requires_grad=True)

@torch.compile(backend="aot_eager", fullgraph=True)
def fn(x):
    return MyFunction.apply(x)

y = fn(x)
print(y)
y[0].backward()
print(x.grad)

ctx