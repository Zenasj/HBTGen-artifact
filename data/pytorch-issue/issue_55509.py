import torch
import torch.nn as nn
from torch.autograd import Function


class PassThrough(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        print(f"grad_outputs {grad_outputs}")
        return grad_outputs


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        ret = a, b
        print(f"In fwd pass {ret[0].requires_grad}")
        ret = PassThrough.apply(ret)
        print(f"After passthrough {ret[0].requires_grad}")
        return ret


model = MyModel()
inp = torch.ones(1)
out = model(inp)
loss = out[0] + out[1]
loss.backward()

import torch
import torch.nn as nn
from torch.autograd import Function


class PassThrough(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        print(f"grad_outputs {grad_outputs}")
        return grad_outputs


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        ret = a
        print(f"In fwd pass {ret.requires_grad}")
        ret = PassThrough.apply(ret)
        print(f"After passthrough {ret.requires_grad}")
        return ret


model = MyModel()
inp = torch.ones(1)
out = model(inp)
out.sum().backward()

xs = [[torch.tensor(1), torch.tensor(2)], (torch.tensor(3),)]
tensors_in_xs, _ = tree_flatten(xs)