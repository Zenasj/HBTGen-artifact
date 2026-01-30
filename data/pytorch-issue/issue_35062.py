import torch
a = torch.rand(0, 4)
print(a.max(1))
print(a.min(1))

torch.return_types.max(
values=tensor([]),
indices=tensor([], dtype=torch.int64))
torch.return_types.max(
values=tensor([]),
indices=tensor([], dtype=torch.int64))

import torch
a = torch.rand(0, 4)
torch.from_numpy(a.max(1).values.numpy())

import torch
a = torch.rand(0, 1)

torch = a.max() # not specify a dimension
# output:  RuntimeError: operation does not have an identity.

torch =  a.max(1)
# output:  torch.return_types.max(values=tensor([]),indices=tensor([], dtype=torch.int64))

torch = a.max(3)
# output: IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 3)

torch = a.max(120)
# output: IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 120)