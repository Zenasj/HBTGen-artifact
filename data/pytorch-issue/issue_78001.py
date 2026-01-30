import torch as pt

# this crashes
print(pt.ones(100, 100, device='mps').nonzero())
# this works
print(pt.ones(100, 100, device='mps').nonzero().contiguous())