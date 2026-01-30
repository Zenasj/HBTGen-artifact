import torch
from torch import autograd

class BadCustomFunction(autograd.Function):
  @staticmethod
  def forward(ctx, inp):
    interm = inp * 2
    ctx.foo = interm
    res = interm ** 2
    return res

  @staticmethod
  def backward(ctx, gres):
      grad = 2 * 2 * ctx.foo * gres
      return grad


inp = torch.rand(2, dtype=torch.double, requires_grad=True)
# Gradcheck is correct
autograd.gradcheck(BadCustomFunction.apply, inp)  # True
# Double grad is silently wrong
autograd.gradgradcheck(BadCustomFunction.apply, inp)  # False