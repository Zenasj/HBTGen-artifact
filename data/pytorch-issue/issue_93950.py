import torch.nn as nn

import torch

import torch.nn.functional as F

weights = torch.rand(10, 3, dtype=torch.float64)
# Doesn't matter whether you zero out here
weights[0, :].zero_()
weights = weights.requires_grad_(True)
input = torch.tensor([[0, 2, 0, 5]])

def fn(weights):
  # The numerical gradient is technically always wrong
  weights = weights.clone()
  weights[0, :].zero_()
  return F.embedding(input, weights, padding_idx=0)

torch.autograd.gradcheck(fn, (weights,))