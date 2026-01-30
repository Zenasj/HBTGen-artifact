import torch
from typing import *

def f(a : Dict[str, Union[torch.Tensor, int]]):
  return a['foo'][..., 1:] + 2

fn = torch.jit.script(f)
print(fn({'foo':torch.Tensor(1,5)}))