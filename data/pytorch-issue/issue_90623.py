import torch
from torch.distributions import Normal

def fn(inputs):
    normal = Normal(inputs, 1)
    return normal.log_prob(inputs)

a = torch.tensor([1])
compiled = torch.compile(fn)
compiled(a)

torch.get_default_dtype