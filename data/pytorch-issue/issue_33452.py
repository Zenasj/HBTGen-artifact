import torch.nn as nn

_im2col = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
_col2im = nn.Fold(top_shape, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

import torch
from torch import optim
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F

torch.manual_seed(1)

x = torch.randn(2, 3, 8, 8, requires_grad=True)
unfold = nn.Unfold(kernel_size=(3, 3))
x_unfold = unfold(x)
y = x_unfold.view(2, -1).sum()
grads = grad(y, x, grad_outputs=torch.ones(y.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]

print(x_unfold.grad_fn, grads.grad_fn)

grads1 = grads + 1
grads1.backward()