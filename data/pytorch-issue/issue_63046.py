import torch

def ident(x):
  return x

@torch.no_grad()
def test(x):
  return ident(x)

j = torch.jit.script(test)