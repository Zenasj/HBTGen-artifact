import torch

value = 0.5

@torch.jit.script
def test(x):
  x[x>0.5] = value 
  # The following doesn't cause error
  # x[x>0.5] = 0.5
  return x