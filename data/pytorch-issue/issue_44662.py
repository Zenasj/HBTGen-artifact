import torch
import torch.nn as nn

x = torch.tensor([
    [[[1., 0., -1., 0.],
      [2., 1., 1., -2.],
      [-1., -2., -3., -4.]],
     [[3., 1., 1., 7.],
      [6., 5., 21., 3.],
      [1., 2., 3., 4.]]]
], requires_grad=True)

def f(x):
    output_size = (1, 2)
    return torch.nn.functional.adaptive_avg_pool2d(x, output_size).sum()

torch.autograd.gradcheck(f, x)