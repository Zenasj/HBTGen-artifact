import torch
S = 10
x = torch.rand(S) # float

y = torch.zeros(S) # float
y[:] = x[:] # float assignment works correctly
print(y.tolist())

y = torch.zeros(S, dtype=torch.long) # long
y[:] = x[:] # assignment from long tensor to float tensor silently fails.
print(y.tolist())

import torch
S = 10
x = torch.rand(S) * 10 # float

y = torch.zeros(S) # float
y[:] = x[:] # float assignment works correctly
print(y.tolist()) #e.g. [9.241122245788574, 6.167181968688965, 7.978866100311279, ...]

y = torch.zeros(S, dtype=torch.long) # long
y[:] = x[:]
print(y.tolist())  # e.g. [9, 6, 7, 9, 5, 7, 0, 0, 4, 6]