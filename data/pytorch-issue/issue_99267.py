import torch.nn as nn

import torch
from torch.func import jacrev, functional_call
inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())
jacobians = jacrev(functional_call, argnums=1)(model, params, (inputs,))

import torch
from torch.func import jacrev, functional_call
inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())
def f(params, inputs):
  return functional_call(model, params, (inputs,))

jacobians = jacrev(f)(params, inputs)