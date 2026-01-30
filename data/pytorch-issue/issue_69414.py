import torch
tensor = torch.rand(torch.Size([4]))
r = 0
import itertools
res1 = list(itertools.combinations(arg_1.tolist(), r=r))
print(res1)
# [()]
res2 = torch.combinations(tesnor, r=r)
# RuntimeError: Expect a positive number, but got 0