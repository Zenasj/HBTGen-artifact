import torch.nn as nn

import torch
import functorch

def poc5():
  device = 'cpu'

  t1 = torch.zeros(50, device=device)
  t1_slice = t1.data[:5]
  # Assigning the view back to origonal tensor's data should be OK.
  t1.data = t1_slice
  print(t1)

print("CPU")
poc5()
print()

print("CPU Functionalize")
functorch.functionalize(poc5)()

3
import torch

x = torch.nn.Parameter(torch.ones(10))
print("x._version before:", x._version)

with torch.no_grad():
    x.copy_(2.0)

print("x._version after:", x._version)

3
import torch

size = 10
param = torch.nn.Parameter(torch.ones(size))
loss = torch.sum(torch.abs(param))

# First, free the underlying storage of `param` without messing with autograd
with torch.no_grad():
    # this doesn't exist today, but pretend that it does and it deallocates param's storage
    # under torch.no_grad, param's version counter will not change
    param.storage().resize_(0)

# Do something (e.g. other computation that takes a lot of memory)

# Later, reconstruct the parameter (without affecting it's autograd data)
# under torch.no_grad, param._version will be unchanged
with torch.no_grad():
    correct_data = torch.ones(size)
    param.storage().resize_(size)
    # this line below causes the error, but `param.data = correct_data` is OK
    param.copy_(correct_data)

loss.backward()