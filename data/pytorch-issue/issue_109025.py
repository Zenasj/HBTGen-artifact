import torch.nn as nn

import torch
from torch import nn
from torch._dynamo import allow_in_graph
from functools import wraps

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

x = torch.randn(3, 5, 3)

func = lambda x: x.t()
# This works
torch.vmap(func)(x)

# This does not work
torch.compile(traceable(torch.vmap(func)))(x)

import torch
from torch import nn
from torch._dynamo import allow_in_graph
from functools import wraps
from torch.func import stack_module_state, functional_call
import functorch
import copy

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

x = torch.randn(3)

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(3, 5)

    def forward(self, x):
        return self.fc(x)


# Following the same recipe as https://pytorch.org/tutorials/intermediate/ensembling.html
models = [Net() for _ in range(5)]
params, buffers = stack_module_state(models)

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')

def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))
vmapped_fmodel = functorch.vmap(fmodel, in_dims=(0, 0, None))

# Works
vmapped_fmodel(params, buffers, x)

# Doesn't work
torch.compile(traceable(vmapped_fmodel))(params, buffers, x)