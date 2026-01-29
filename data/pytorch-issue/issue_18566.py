# torch.rand(2, 1, dtype=torch.float32)
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

timeline = []

class Log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        timeline.append(f"{name}:forward")
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        timeline.append(f"{name}:backward")
        return grad_output, None

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        a, b = x[0], x[1]
        a = Log.apply(a, 'a')
        b = checkpoint(lambda b: Log.apply(b, 'b'), b)
        out = torch.cat((a, b)).sum()
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1, requires_grad=True)

