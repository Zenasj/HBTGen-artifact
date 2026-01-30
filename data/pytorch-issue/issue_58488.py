import torch
import torch.nn as nn

class Sum(nn.Module):
  def forward(self, X, Y):
    return X + Y
  def right_inverse(Z):
    return Z, torch.zeros_like(Z)

X = module.weight
Y = param.right_inverse(X)
assert isinstance(Y, Tensor) or isinstance(Y, collections.Sequence)
Z = param(Y) if isisntance(Y, Tensor) else param(*Y)