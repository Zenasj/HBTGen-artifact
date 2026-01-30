import torch

@torch.compile()
def version_bump_test(x):
    x[...] = 0

x = torch.ones(4)
print("v0", x._version)  # Expected: 0
version_bump_test(x)
print("v1", x._version)  # Expected: 1, Actual: 0
print(x)  # Expected: tensor([0., 0., 0., 0.])

import torch

@torch.compile()
def version_bump_test(x):
    x[...] = 0

x = torch.ones(4, requires_grad = True)
version_bump_test(x)