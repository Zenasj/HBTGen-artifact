import torch

op = torch.Tensor.acos_

def fn(z):
  x = z.clone()
  result = torch.vmap(op)(x)
  assert(x is result)  # Fails in nopython=True when `is_` is implemented as testing equality of fake tensor
  
torch.compile(fn)(torch.zeros(10))