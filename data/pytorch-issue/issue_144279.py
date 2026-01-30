import torch

from torch.distributions import *
p = Uniform(0,1)
q = Beta(1,1)  # special case, equal to Uniform(0,1)
kl_divergence(p, q)  # tensor(nan) ❌
kl_divergence(p, p)  # tensor(0) ✅