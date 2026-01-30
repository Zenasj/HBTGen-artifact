import torch
from functorch import make_fx, grad, vmap, functionalize

def f(x):  # tests that inputs are still successfully mutated
    tmp = torch.ones(4)
    y = x.view(4)
    y.add_(tmp)
    return x

def f(x):  # test the free variable mutation case, which currently breaks in functorch
    tmp = torch.ones(4)
    tmp.add_(x)
    return tmp

batched_input = torch.ones(2, 4)
vmap(functionalize((f)))(batched_input)
vmap(functionalize((f2)))(batched_input)

import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
a = torch.ones(2, 2, device=device)
b = a.view(4)
a.add_(1) # successfully mutates b