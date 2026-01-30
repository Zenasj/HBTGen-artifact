import torch

# For reference we get the result on cpu
a = torch.ones(3).to('cpu')
a[1].zero_()
print(a) # -> tensor([1., 0., 1.])

# Now we do the same on MPS
b = torch.ones(3).to('mps')
b[1].zero_()
print(b) # -> tensor([1., 0., 0.], device='mps:0')