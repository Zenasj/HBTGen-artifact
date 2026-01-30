import torch.nn as nn

import torch

# Normal Tensor
t = torch.nn.parameter.Parameter(torch.Tensor())
print(t)
print('UninitializedParameter?', isinstance(t, torch.nn.parameter.UninitializedParameter))
# Parameter containing:
# tensor([], requires_grad=True)
# UninitializedParameter? False

# Custom tensor
class MyTensor(torch.Tensor):
  pass

mt = torch.nn.parameter.Parameter(MyTensor())
print(mt, isinstance(mt, torch.nn.parameter.UninitializedParameter))
print('UninitializedParameter?', isinstance(mt, torch.nn.parameter.UninitializedParameter))
# Parameter(MyTensor([], requires_grad=True)) True
# UninitializedParameter? True