import torch

py
sampler = RandomSampler(ds)
torch.manual_seed(0)
l1 = list(sampler)
torch.manual_seed(0)
l2 = list(sampler)
# Expect same
assert l1 == l2