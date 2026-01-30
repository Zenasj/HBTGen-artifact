import torch

def foo(a, b):
  c = a.mul(b)
  a = c.mul(c)
  a = c.mul(a)
  return a

@torch.jit.script
def foo_compiled(a, b):
  if a.size(0) == b.size(0):
    c = a.mul(b)
    a = c.mul(c)
    a = c.mul(a)
  return a