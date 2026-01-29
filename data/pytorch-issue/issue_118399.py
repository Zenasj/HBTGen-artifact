# torch.rand(1, dtype=torch.float32)  # Inferred input shape from the provided tensor

import torch
from torch import nn

foo = []

class MulY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x * 3

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.stride(0) == 1:
            foo.append(grad_out)
            return grad_out * 2
        return grad_out * 3

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return MulY.apply(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor(2., requires_grad=True)

