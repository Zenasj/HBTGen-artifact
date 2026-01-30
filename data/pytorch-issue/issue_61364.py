import torch

@torch.jit.script
def foo(x: int, y: int):
     z = x + y
     if x == y:
          z = x + y
     return z