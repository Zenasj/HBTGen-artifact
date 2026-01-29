# torch.rand(B=1, C=2, H=1, W=1, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn
from torch.autograd import Function
import torch._dynamo

@torch._dynamo.allow_in_graph
class Foo(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x0 = x.size(0)
        return x * 2

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.x0

class MyModel(nn.Module):
    def forward(self, x):
        return Foo.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, 1, dtype=torch.float32, requires_grad=True)

