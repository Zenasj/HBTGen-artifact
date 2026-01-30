import torch
import numpy as np
a = torch.from_numpy(np.load("a.npy"))
b = torch.from_numpy(np.load("b.npy"))
b = b@b.T

def norm(m):
  return torch.mean(m**2)

def check(m1, m2):
  print(f"m1 norm: {norm(m1)} m2 norm: {norm(m2)}, difference norm: {norm(m1-m2)}")
  # print(m1-m2)

print(a.shape, b.shape)

# on cpu everything is more or less OK (error is on the order of float precision)
check(a.T@b, (b.T@a).T)

# subset of b columns - same
check(a.T@b[:,:10], (b[:,:10].T@a).T)

# on cuda however...
a = a.cuda()
b = b.cuda()
check(a.T@b, (b.T@a).T)

# also on cuda, this time it's OK
check(a.T@b[:,:10], (b[:,:10].T@a).T)