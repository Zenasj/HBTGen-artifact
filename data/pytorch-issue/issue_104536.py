def get_div_fn(f):

    @torch._dynamo.allow_in_graph
    def _f(x, t):
        return torch.trace(jacrev(f)(x,t))

    return lambda x, t : _f(x, t)

import torch
from torch.func import jacrev
import torch._dynamo
torch._dynamo.config.verbose = True

def get_div_fn(f):

    @torch._dynamo.allow_in_graph
    def _f(x, t):
        return torch.trace(jacrev(f)(x,t))

    return lambda x, t : _f(x, t)

# example function
f = lambda x, t: t * 3 * x

# example inputs
x = torch.tensor([1.,2.,3.])
t = torch.tensor([0.5])

# expected answer is 4.5
divf = get_div_fn(f)
divf_compiled = torch.compile(divf)
print(divf(x,t)) # works
print(divf_compiled(x,t)) # breaks

def get_div_fn(f):
 
    @torch._dynamo.allow_in_graph
    def _f(x, t):
        return torch.trace(jacrev(f)(x,t))
 
    return lambda x, t : _f(x, t)

divf = get_div_fn(f)
divf_compiled = torch.compile(divf)