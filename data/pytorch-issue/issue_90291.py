import torch

t = torch.tensor([1, 2, 3])
print(torch.slice_copy(t, dim=0, step=1)) # [1, 2, 3]
# the default value of `step` is 1 in slice_copy
print(torch.slice_copy(t, dim=0)) # [1]

# fails
assert torch.equal(torch.slice_copy(t, dim=0, step=1), torch.slice_copy(t, dim=0))