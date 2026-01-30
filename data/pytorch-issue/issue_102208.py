import torch
from functools import partial

batch_lin = torch.vmap(partial(torch.linspace, steps = 10))
start = torch.tensor([1.,2.,3.])
stop = torch.tensor([25.,26.,27.])
print(batch_lin(start, stop))