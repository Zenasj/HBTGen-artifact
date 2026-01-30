import torch

def f(x):
    return x[x > 0]

jf = torch.jit.trace(f, torch.tensor(2., device="cuda"))

import torch

def f(x):
    return torch.distributions.HalfCauchy(x).log_prob(x)

jf = torch.jit.trace(f, torch.tensor(2., device="cuda"))