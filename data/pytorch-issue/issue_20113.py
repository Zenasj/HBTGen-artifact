import torch

@torch.jit.script
def norm_test():
  t = torch.ones(10, 5)
  return torch.norm(t, dim=1, keepdim=True)

import torch

@torch.jit.script
def norm_test():
  t = torch.ones(10, 5)
  return torch.norm(t, p=2, dim=1)

@torch.jit.script
def norm_test():
  t = torch.ones(10, 5)
  return t.norm(p="fro", dim=1)

@torch.jit.script
def norm_test():
  t = torch.ones(10, 5)
  return t.norm(dim=1)