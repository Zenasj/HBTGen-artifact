import torch

@torch.jit.script
def test():
  li = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
  return torch.tensor(li)