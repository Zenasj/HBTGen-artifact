import torch

print(torch.__version__) # should print local version (not release)

a1 = torch.tensor([1,2,3])
a2 = torch.tensor([2])

a1 @ a2