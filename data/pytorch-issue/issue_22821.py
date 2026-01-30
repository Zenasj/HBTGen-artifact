import torch
import torch.nn as nn

t = torch.ones(2, 3)
v = torch.autograd.Variable(t).requires_grad_()
y = v * v
t.add_(1)  # This bumps version counter of `t`
y.sum().backward()  # This computes `v`'s gradient incorrectly before this patch, and throws error after this patch

t = torch.ones(2, 3)
v = torch.nn.Parameter(t)
y = v * v
t.add_(1)  # This bumps version counter of `t`
y.sum().backward()  # This computes `v`'s gradient incorrectly before this patch, and throws error after this patch