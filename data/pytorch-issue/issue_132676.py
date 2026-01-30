import torch

@torch.compile
def to_float(x):
  return x.to(torch.float)