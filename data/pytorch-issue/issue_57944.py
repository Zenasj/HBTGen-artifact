import torch
import torch.nn as nn

from torch import nn, Tensor
from torch.autograd import Variable


class MultiplyModule(nn.Module):
    def forward(self, x1, x2):
        return x1 * x2


t1 = Variable(Tensor([3]), requires_grad=False)
t2 = Variable(Tensor([5]), requires_grad=True)

add_module = MultiplyModule()
add_module.register_full_backward_hook(lambda m, x, y: print('In backward hook', x, y))

t3 = add_module(t1, t2)
t3.backward()